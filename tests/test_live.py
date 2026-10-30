import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from certiroute.config import load_config
from eval import live_study
from eval.benchmarks import BenchmarkRow
from eval.live import Budget, MeteredTransport, summarize_calls


def mock_spec():
    spec = json.loads(Path("configs/live.study.json").read_text())
    # Test-only rates; never assert that these are xAI account prices.
    spec["strong"].update(input=4.0, cached_input=4.0, output=8.0)
    return spec


@pytest.mark.parametrize("limit", [3, 18, 20])
def test_old_budget_limits_cannot_be_reenabled(tmp_path, limit):
    with pytest.raises(ValueError, match="budget"):
        Budget(tmp_path / "journal", limit)
    spec = mock_spec()
    spec["budget_usd"] = limit
    with pytest.raises(ValueError, match="authorization"):
        live_study.validate_spec(spec)


def test_xai_billed_ticks_and_output_cap(tmp_path):
    async def run():
        def upstream(request):
            body = json.loads(request.content)
            assert body["max_completion_tokens"] == 32
            assert "max_tokens" not in body
            assert "provider" not in body
            return httpx.Response(
                200,
                json={
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "cost_in_usd_ticks": 100000,
                        "cost": 0,
                    }
                },
            )

        meter = MeteredTransport(
            httpx.MockTransport(upstream),
            Budget(tmp_path / "journal", 2.5),
            {
                "api.x.ai": {
                    "tier": "strong",
                    "model": "grok-4.20-0309-non-reasoning",
                    "input": 1.25,
                    "output": 2.5,
                }
            },
        )
        async with httpx.AsyncClient(transport=meter) as client:
            await client.post(
                "https://api.x.ai/v1/chat/completions",
                json={"model": "grok-4.20-0309-non-reasoning", "messages": [], "max_tokens": 32},
            )
        assert meter.calls[0]["cost_usd"] == pytest.approx(0.00001)
        assert meter.calls[0]["cost_basis"] == "xai_reported_usd_ticks"

    asyncio.run(run())


def test_current_model_requires_confirmed_prices():
    spec = json.loads(Path("configs/live.study.json").read_text())
    assert spec["strong"]["model"] == "grok-4.20-0309-non-reasoning"
    spec["strong"]["input"] = None
    with pytest.raises(ValueError, match="prices"):
        live_study.validate_spec(spec)


def test_cli_reports_safe_acquisition_failure(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(mock_spec()))
    out = tmp_path / "run"

    async def failed_study(spec, output, budget, pilot_requests=None, **kwargs):
        raise live_study.AcquisitionError("strong", 402, False)

    monkeypatch.setattr(live_study, "study", failed_study)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_study",
            "--execute",
            "--config",
            str(config),
            "--out",
            str(out),
            "--journal",
            str(tmp_path / "budget.jsonl"),
        ],
    )
    with pytest.raises(SystemExit, match="xAI.*HTTP 402"):
        live_study.main()
    result = json.loads((out / "results.json").read_text())
    assert result["failure"]["provider"] == "xAI"
    assert result["failure"]["http_status"] == 402
    assert "credit" in result["failure"]["hint"].lower()
    assert not (tmp_path / "budget.lock").exists()


def test_budget_survives_interruption_and_rejects_overspend(tmp_path):
    path = tmp_path / "budget.jsonl"
    budget = Budget(path, 0.10)
    ident = budget.reserve(0.07, {"tier": "cheap"})
    assert Budget(path, 0.10).charged == pytest.approx(0.07)
    with pytest.raises(ValueError, match="budget"):
        Budget(path, 0.10).reserve(0.04, {})
    budget.settle(ident, 0.02, {})
    assert Budget(path, 0.10).charged == pytest.approx(0.02)
    with pytest.raises(ValueError):
        budget.reserve(-1, {})


def test_pilot_cap_preserves_cumulative_spend(tmp_path):
    path = tmp_path / "budget.jsonl"
    old = Budget(path, 2.5)
    ident = old.reserve(0.1, {})
    old.settle(ident, 0.1, {})
    pilot = Budget(path, 2.5, run_limit=0.5)
    pilot.reserve(0.4, {})
    with pytest.raises(ValueError, match="budget"):
        pilot.reserve(0.11, {})
    assert Budget(path, 2.5).charged == pytest.approx(0.5)


def test_xai_does_not_treat_unverified_cost_field_as_usd(tmp_path):
    async def run():
        meter = MeteredTransport(
            httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0}}
                )
            ),
            Budget(tmp_path / "journal", 1),
            {
                "api.x.ai": {
                    "tier": "strong",
                    "model": "grok-4.20-0309-non-reasoning",
                    "input": 4,
                    "output": 8,
                }
            },
        )
        async with httpx.AsyncClient(transport=meter) as client:
            await client.post(
                "https://api.x.ai/v1/chat/completions",
                json={"model": "grok-4.20-0309-non-reasoning", "max_tokens": 10, "messages": []},
            )
        assert meter.calls[0]["cost_usd"] == pytest.approx(56e-6)
        assert meter.calls[0]["cost_basis"] != "provider_reported"

    asyncio.run(run())


def test_full_cli_requires_account_price_confirmation(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(mock_spec()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_study",
            "--execute",
            "--config",
            str(config),
            "--mode",
            "full",
            "--pilot-report",
            str(tmp_path / "pilot.json"),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit):
        live_study.main()
    assert not (tmp_path / "out").exists()


def test_full_study_requires_matching_successful_pilot():
    spec = json.loads(Path("configs/live.study.json").read_text())
    with pytest.raises(ValueError, match="pilot"):
        live_study.validate_pilot_report({}, spec, 0)


def test_xai_preflight_accepts_model_catalog_without_openrouter_prices(monkeypatch):
    spec = json.loads(Path("configs/live.study.json").read_text())
    spec["pricing_as_of"] = live_study.date.today().isoformat()
    requests = []

    def upstream(request):
        requests.append(request)
        tier = "cheap" if request.url.host == "api.groq.com" else "strong"
        assert request.method == "GET" and request.url.path.endswith("/models")
        assert request.headers["Authorization"] == "Bearer test-" + tier
        return httpx.Response(200, json={"data": [{"id": spec[tier]["model"]}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    asyncio.run(live_study.preflight(spec, {"cheap": "test-cheap", "strong": "test-strong"}))
    assert len(requests) == 2
    assert requests[1].url.host == "api.x.ai"


@pytest.mark.parametrize(
    "url",
    [
        "https://openrouter.ai/api/v1",
        "http://api.x.ai/v1",
        "https://api.x.ai.evil.test/v1",
    ],
)
def test_xai_rejects_wrong_endpoint(url):
    spec = json.loads(Path("configs/live.study.json").read_text())
    spec["strong"]["base_url"] = url
    with pytest.raises(ValueError, match="endpoint"):
        live_study.validate_spec(spec)


def test_transport_records_usage_but_never_credentials(tmp_path):
    async def run():
        journal = tmp_path / "journal.jsonl"
        budget = Budget(journal, 1)
        upstream = httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "model": "cheap",
                    "choices": [{"message": {"content": "2"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            )
        )
        meter = MeteredTransport(
            upstream,
            budget,
            {
                "cheap.test": {
                    "tier": "cheap",
                    "model": "cheap",
                    "input": 1,
                    "output": 2,
                }
            },
        )
        async with httpx.AsyncClient(transport=meter) as client:
            response = await client.post(
                "https://cheap.test/v1/chat/completions",
                headers={"Authorization": "Bearer secret-value"},
                json={
                    "model": "cheap",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert response.status_code == 200
        assert meter.calls[0]["usage"]["prompt_tokens"] == 10
        assert meter.calls[0]["cost_usd"] == pytest.approx(14e-6)
        assert "secret-value" not in journal.read_text()

    asyncio.run(run())


def test_latency_and_errors_include_all_attempts():
    summary = summarize_calls(
        [
            {"ok": True, "latency_ms": 10, "cost_usd": 0.01, "cost_known": True},
            {"ok": False, "latency_ms": 100, "cost_usd": 0.03, "cost_known": False},
        ]
    )
    assert summary["error_rate"] == 0.5
    assert summary["unknown_cost_requests"] == 1
    assert summary["cost_usd"] == pytest.approx(0.04)
    assert summary["latency_p95_ms"] >= summary["latency_p50_ms"]


@pytest.mark.parametrize(
    "overcharge,pilot,resume",
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True)],
)
def test_complete_live_workflow_with_mock_providers(
    tmp_path, monkeypatch, overcharge, pilot, resume
):
    spec = mock_spec()
    spec.update(train=20, calibration=20, test=20, interval_seconds=0)
    rows = [
        BenchmarkRow(
            id=f"case-{i}",
            task="math",
            expected="7",
            q_cheap=0,
            q_strong=0,
            request={"messages": [{"role": "user", "content": f"Compute case {i}: return seven."}]},
            cheap_output="",
            strong_output="",
            cheap_usage={"prompt_tokens": 0, "completion_tokens": 0},
            strong_usage={"prompt_tokens": 0, "completion_tokens": 0},
        )
        for i in range(60)
    ]
    source_config = load_config(__import__("pathlib").Path("configs/offline.gsm8k.yaml"))
    monkeypatch.setattr(
        live_study, "load_task", lambda *args: (rows, {"dataset_sha256": "a" * 64}, source_config)
    )

    async def preflight(*args):
        pass

    monkeypatch.setattr(live_study, "preflight", preflight)
    monkeypatch.setenv(spec["cheap"]["api_key_env"], "test-cheap-secret")
    monkeypatch.setenv(spec["strong"]["api_key_env"], "test-strong-secret")

    call_count = 0

    def upstream(request):
        nonlocal call_count
        call_count += 1
        if resume and call_count == 4:
            return httpx.Response(401, json={})
        body = json.loads(request.content)
        index = int(body["messages"][0]["content"].split()[2].rstrip(":"))
        text = "7" if body["model"] == spec["strong"]["model"] or index % 2 else "8"
        usage = {"prompt_tokens": 10, "completion_tokens": 2}
        if overcharge and call_count == 140:
            usage["prompt_tokens"] = 1000000
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": text}}],
                "usage": usage,
            },
        )

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(upstream))
    output = tmp_path / "run"
    output.mkdir()
    if resume:
        with pytest.raises(live_study.AcquisitionError):
            asyncio.run(live_study.study(spec, output, Budget(tmp_path / "budget.jsonl", 2.5)))
        assert call_count == 4
        (output / "results.json").write_text(json.dumps({"status": "incomplete"}))
    if overcharge:
        with pytest.raises(ValueError, match="halt"):
            asyncio.run(live_study.study(spec, output, Budget(tmp_path / "budget.jsonl", 2.5)))
        assert call_count == 140
        assert Budget(tmp_path / "budget.jsonl", 2.5).halted
        return
    result = asyncio.run(
        live_study.study(
            spec,
            output,
            Budget(tmp_path / "budget.jsonl", 2.5),
            pilot_requests=20 if pilot else None,
            resume=resume,
        )
    )
    if pilot:
        assert result["status"] == "pilot_complete"
        assert result["upstream_attempts"]["requests"] == call_count == 20
        assert result["summaries"]["strong-only"]["requests"] == 10
        assert result["eligible_for_full_study"] is True
        assert not (output / "calibration.json").exists()
        assert not (output / "router.json").exists()
        assert result["projected_full_study_usd"] > 0
        live_study.validate_pilot_report(
            result, spec, result["budget_charged_usd_including_prior_runs"]
        )
        with pytest.raises(ValueError, match="pilot"):
            live_study.validate_pilot_report(result, {**spec, "max_tokens": 512}, 0)
        with pytest.raises(ValueError, match="budget"):
            live_study.validate_pilot_report(result, spec, 2.5)
        return
    assert result["status"] == "complete"
    assert result["upstream_attempts"]["requests"] == call_count == 140 + int(resume)
    if resume:
        assert result["execution"]["reused_acquisition_responses"] == 3
        assert result["acquisition"]["requests"] == 81
        assert result["acquisition"]["unknown_cost_requests"] == 1
    assert result["summaries"]["CertiRoute"]["requests"] == 20
    assert result["summaries"]["strong-only"]["quality"] == 1
    assert result["summaries"]["CertiRoute"]["regret_denominator"] == 20
    saved = json.dumps(result) + (tmp_path / "budget.jsonl").read_text()
    assert "test-cheap-secret" not in saved and "test-strong-secret" not in saved


def test_missing_usage_charges_reservation(tmp_path):
    async def run():
        budget = Budget(tmp_path / "journal.jsonl", 0.01)
        meter = MeteredTransport(
            httpx.MockTransport(lambda r: httpx.Response(200, json={})),
            budget,
            {"cheap.test": {"tier": "cheap", "model": "cheap", "input": 1, "output": 2}},
        )
        async with httpx.AsyncClient(transport=meter) as client:
            await client.post(
                "https://cheap.test/chat/completions",
                json={"model": "cheap", "max_tokens": 10, "messages": []},
            )
        assert not meter.calls[0]["cost_known"]
        assert budget.charged > 0.001

    asyncio.run(run())


def test_openrouter_requests_enforce_recorded_price_caps(tmp_path):
    async def run():
        def upstream(request):
            body = json.loads(request.content)
            assert body["provider"]["max_price"] == {"prompt": 2.5, "completion": 10}
            assert body["provider"]["require_parameters"] is True
            return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

        meter = MeteredTransport(
            httpx.MockTransport(upstream),
            Budget(tmp_path / "journal", 1),
            {
                "openrouter.ai": {
                    "tier": "strong",
                    "model": "openai/gpt-4o",
                    "input": 2.5,
                    "output": 10,
                },
            },
        )
        async with httpx.AsyncClient(transport=meter) as client:
            await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o",
                    "max_tokens": 10,
                    "messages": [],
                },
            )

    asyncio.run(run())


def test_unexpected_charge_stops_later_requests_even_after_restart(tmp_path):
    path = tmp_path / "journal"
    budget = Budget(path, 1)
    ident = budget.reserve(0.01, {})
    with pytest.raises(ValueError, match="reservation"):
        budget.settle(ident, 0.03, {})
    assert Budget(path, 1).charged == 0.03
    with pytest.raises(ValueError, match="halted"):
        Budget(path, 1).reserve(0.01, {})


@pytest.mark.parametrize("cached,known,expected", [(8, True, 10e-6), (11, False, None)])
def test_cached_usage_is_priced_and_invalid_counts_stay_reserved(tmp_path, cached, known, expected):
    async def run():
        meter = MeteredTransport(
            httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    json={
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "prompt_tokens_details": {"cached_tokens": cached},
                        },
                    },
                )
            ),
            Budget(tmp_path / "journal", 1),
            {
                "cheap.test": {
                    "tier": "cheap",
                    "model": "cheap",
                    "input": 1,
                    "output": 2,
                    "cached_input": 0.5,
                }
            },
        )
        async with httpx.AsyncClient(transport=meter) as client:
            await client.post(
                "https://cheap.test/chat/completions",
                json={
                    "model": "cheap",
                    "max_tokens": 10,
                    "messages": [],
                },
            )
        assert meter.calls[0]["cost_known"] is known
        if known:
            assert meter.calls[0]["cost_usd"] == pytest.approx(expected)
        else:
            assert meter.calls[0]["cost_usd"] > 0.001

    asyncio.run(run())
