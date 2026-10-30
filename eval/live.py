"""Metered live study primitives. Secrets never enter the journal or report."""

import json
import math
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx

from eval.repeated import quantile


class Budget:
    """Append-only reservation journal; interrupted attempts retain their full reservation.

    The command takes a process lock before using this object. One sequential study
    owns the ledger; callers must not instantiate concurrent writers.
    """

    def __init__(self, path: Path, limit: float, run_limit: float | None = None) -> None:
        if not math.isfinite(limit) or not 0 < limit <= 2.5:
            raise ValueError(
                "API budget must be positive and at most $2.50 (retain $0.49 for fees)"
            )
        if run_limit is not None and (not math.isfinite(run_limit) or not 0 < run_limit <= limit):
            raise ValueError("Invalid per-run budget")
        self.run_limit = run_limit
        self.path, self.limit = path, limit
        self.entries: dict[str, float] = {}
        self.settled: set[str] = set()
        self.halted = False
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["event"] == "reserve":
                    self.entries[event["id"]] = event["usd"]
                elif event["event"] == "settle":
                    self.entries[event["id"]] = event["usd"]
                    self.settled.add(event["id"])
                    self.halted = self.halted or event.get("reservation_exceeded", False)
        self.initial_charged = self.charged

    @property
    def charged(self) -> float:
        return math.fsum(self.entries.values())

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, usd: float, metadata: dict[str, Any]) -> str:
        if self.halted:
            raise ValueError("Live budget halted after an unexpected provider charge")
        if not math.isfinite(usd) or usd <= 0:
            raise ValueError("Invalid reservation")
        if self.charged + usd > self.limit:
            raise ValueError("Live budget exhausted before sending request")
        if (
            self.run_limit is not None
            and self.charged - self.initial_charged + usd > self.run_limit
        ):
            raise ValueError("Pilot budget exhausted before sending request")
        ident = uuid.uuid4().hex
        self.append({**metadata, "event": "reserve", "id": ident, "usd": usd})
        self.entries[ident] = usd
        return ident

    def settle(self, ident: str, usd: float, metadata: dict[str, Any]) -> None:
        if ident not in self.entries or ident in self.settled or not math.isfinite(usd) or usd < 0:
            raise ValueError("Invalid settlement")
        exceeded = usd > self.entries[ident] + 1e-12
        self.append(
            {
                **metadata,
                "event": "settle",
                "id": ident,
                "usd": usd,
                "reservation_exceeded": exceeded,
            }
        )
        self.entries[ident] = usd
        self.settled.add(ident)
        if exceeded:
            self.halted = True
            raise ValueError("Provider charge exceeded its reservation; study halted")


class MeteredTransport(httpx.AsyncBaseTransport):
    """Non-streaming study transport; measures and pays for every fallback attempt."""

    def __init__(
        self,
        upstream: httpx.AsyncBaseTransport,
        budget: Budget,
        providers: dict[str, dict[str, Any]],
    ) -> None:
        self.upstream, self.budget, self.providers = upstream, budget, providers
        self.calls: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        spec = self.providers[request.url.host]
        body = json.loads(request.content)
        if body.get("model") != spec["model"] or body.get("stream"):
            raise ValueError("Unmetered model or streaming request forbidden")
        maximum = body.get("max_tokens")
        if not isinstance(maximum, int) or not 0 < maximum <= 1024:
            raise ValueError("Bounded max_tokens required")
        if request.url.host == "api.x.ai":
            if body["model"] != "grok-4.20-0309-non-reasoning":
                raise ValueError("Budgeted xAI study requires the fixed non-reasoning model")
            body["max_completion_tokens"] = body.pop("max_tokens")
            headers = request.headers.copy()
            headers.pop("content-length", None)
            request = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                json=body,
                extensions=request.extensions,
            )
        if request.url.host == "openrouter.ai":
            # Apply equally to direct strong calls and gateway upstream calls, without
            # adding a provider-specific option to the calibrated gateway request scope.
            body["provider"] = {
                "max_price": {"prompt": spec["input"], "completion": spec["output"]},
                "require_parameters": True,
            }
            headers = request.headers.copy()
            headers.pop("content-length", None)
            request = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                json=body,
                extensions=request.extensions,
            )
        # UTF-8 bytes upper bound ordinary tokenizer input, plus generous chat framing.
        # No tools, images or hidden tool calls are permitted in this study.
        input_bound = len(json.dumps(body["messages"], ensure_ascii=False).encode()) + 1024
        reserve = 1.10 * (input_bound * spec["input"] + maximum * spec["output"]) / 1e6
        ident = self.budget.reserve(reserve, {"tier": spec["tier"], "model": spec["model"]})
        start = time.perf_counter()
        event: dict[str, Any] = {
            "id": ident,
            "tier": spec["tier"],
            "ok": False,
            "cost_usd": reserve,
            "reserved_usd": reserve,
            "cost_known": False,
            "usage": None,
            "started_at": datetime.now(UTC).isoformat(),
            "provider_controls": body.get("provider"),
        }
        try:
            response = await self.upstream.handle_async_request(request)
            await response.aread()
            event["status"] = response.status_code
            event["ok"] = response.is_success
            # Allowlist numeric Retry-After only. Never persist raw response headers.
            try:
                retry_after = float(response.headers.get("retry-after", ""))
                if math.isfinite(retry_after) and retry_after >= 0:
                    event["retry_after_seconds"] = retry_after
            except ValueError:
                pass
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    for field in ("id", "provider"):
                        if isinstance(payload.get(field), str):
                            event["upstream_" + field] = payload[field]
                usage = payload.get("usage") if isinstance(payload, dict) else None
                if isinstance(usage, dict):
                    counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
                    counts_valid = all(
                        isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts
                    )
                    details = usage.get("prompt_tokens_details") or {}
                    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else None
                    cache_valid = (
                        isinstance(cached, int)
                        and not isinstance(cached, bool)
                        and counts_valid
                        and 0 <= cached <= cast(int, counts[0])
                    )
                    if counts_valid and cache_valid:
                        cost = (
                            (cast(int, counts[0]) - cast(int, cached)) * spec["input"]
                            + cast(int, cached) * spec.get("cached_input", spec["input"])
                            + cast(int, counts[1]) * spec["output"]
                        ) / 1e6
                        event["cost_basis"] = (
                            "actual_usage_times_recorded_cache_prices"
                            if "cached_tokens" in details and "cached_input" in spec
                            else "uncached_input_upper_bound"
                        )
                        event.update(
                            cost_usd=cost,
                            cost_known=True,
                            usage=usage,
                            returned_model=payload.get("model"),
                        )
                    billed = usage.get("cost")
                    if (
                        request.url.host == "openrouter.ai"
                        and isinstance(billed, (int, float))
                        and not isinstance(billed, bool)
                        and math.isfinite(billed)
                        and billed >= 0
                    ):
                        event.update(
                            cost_usd=float(billed), cost_known=True, cost_basis="provider_reported"
                        )
                    ticks = usage.get("cost_in_usd_ticks")
                    if (
                        request.url.host == "api.x.ai"
                        and type(ticks) is int
                        and 0 <= ticks <= 10**15
                    ):
                        event.update(
                            cost_usd=ticks / 1e10,
                            cost_known=True,
                            cost_basis="xai_reported_usd_ticks",
                        )
            except (ValueError, TypeError):
                event["ok"] = False
            return response
        finally:
            event["latency_ms"] = 1000 * (time.perf_counter() - start)
            self.calls.append(event)
            self.budget.settle(ident, event["cost_usd"], event)

    async def aclose(self) -> None:
        await self.upstream.aclose()


def summarize_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    if not calls:
        return {"requests": 0, "status": "not_run"}
    latencies = [c["latency_ms"] for c in calls]
    attempts = [a for c in calls for a in c.get("attempts", [c])]
    known_usage = [a["usage"] for a in attempts if a.get("usage") is not None]
    return {
        "requests": len(calls),
        "error_rate": sum(not c["ok"] for c in calls) / len(calls),
        "latency_p50_ms": quantile(latencies, 0.5),
        "latency_p95_ms": quantile(latencies, 0.95),
        "cost_usd": math.fsum(c["cost_usd"] for c in calls),
        "unknown_cost_requests": sum(not c["cost_known"] for c in calls),
        "known_prompt_tokens": sum(u["prompt_tokens"] for u in known_usage),
        "known_completion_tokens": sum(u["completion_tokens"] for u in known_usage),
        "attempts_without_usage": sum(a.get("usage") is None for a in attempts),
    }
