import asyncio
import json
import sys

import httpx
import pytest

from eval import live_study
from eval.live import Budget, MeteredTransport
from eval.live_recovery import ResumeError, acquire_with_retry, load_checkpoint


@pytest.mark.parametrize("status,retry_after,expected", [(429, 1, 2), (401, 1, 1), (429, 301, 1)])
def test_retries_only_short_rate_limits(monkeypatch, status, retry_after, expected):
    calls, waits = [], []

    async def operation():
        calls.append(1)
        return {
            "status": status if len(calls) == 1 else 200,
            "attempts": [{"retry_after_seconds": retry_after}],
        }

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    result = asyncio.run(acquire_with_retry(operation, "cheap", retries=3))
    assert len(calls) == expected
    assert len(waits) == expected - 1
    assert result["status"] == (200 if expected == 2 else status)
    if waits:
        assert waits[0] >= retry_after


def test_retries_exhaust_and_pilot_never_retries(monkeypatch):
    calls, waits = [], []

    async def operation():
        calls.append(1)
        return {"status": 429, "attempts": [{}]}

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(acquire_with_retry(operation, "cheap", retries=3))
    assert len(calls) == 4
    assert sum(waits) == 60 + 120 + 240
    asyncio.run(acquire_with_retry(operation, "cheap", retries=0))
    assert len(calls) == 5


@pytest.mark.parametrize("header,expected", [("12.5", 12.5), ("nan", None), ("secret", None)])
def test_retry_header_logged_as_safe_number_only(tmp_path, header, expected):
    async def run():
        budget = Budget(tmp_path / "journal", 1)
        meter = MeteredTransport(
            httpx.MockTransport(
                lambda r: httpx.Response(
                    429, headers={"retry-after": header, "authorization": "secret"}, json={}
                )
            ),
            budget,
            {"cheap.test": {"tier": "cheap", "model": "cheap", "input": 1, "output": 2}},
        )
        async with httpx.AsyncClient(transport=meter) as client:
            await client.post(
                "https://cheap.test/chat/completions",
                json={"model": "cheap", "messages": [], "max_tokens": 10},
            )
        assert meter.calls[0].get("retry_after_seconds") == expected
        assert "secret" not in budget.path.read_text()
        assert budget.charged > 0  # Unknown usage remains reserved, including HTTP 429.

    asyncio.run(run())


def checkpoint(tmp_path):
    budget = Budget(tmp_path / "journal", 1)
    ident = budget.reserve(0.01, {})
    budget.settle(ident, 0.001, {})
    splits = {
        "mode": "full",
        "train": ["a"],
        "calibration": ["b"],
        "test": ["c"],
        "source_sha256": "source",
        "spec": {"model": "fixed"},
    }
    item = {
        "id": "a",
        "policy": "cheap-only",
        "phase": "acquisition",
        "status": 200,
        "ok": True,
        "cost_known": True,
        "output": "7",
        "cost_usd": 0.001,
        "attempts": [{"id": ident, "tier": "cheap", "cost_usd": 0.001}],
    }
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    (tmp_path / "results.json").write_text(json.dumps({"status": "incomplete"}))
    (tmp_path / "responses.jsonl").write_text(json.dumps(item) + "\n")
    return budget, splits, item


@pytest.mark.parametrize(
    "mutation",
    ["test", "wrong_id", "changed_spec", "empty_journal", "duplicate", "changed_cost", "completed"],
)
def test_resume_rejects_incompatible_evidence(tmp_path, mutation):
    budget, splits, item = checkpoint(tmp_path)
    if mutation == "test":
        item["phase"] = "test"
    elif mutation == "wrong_id":
        item["id"] = "c"
    elif mutation == "changed_spec":
        splits["spec"] = {"model": "other"}
    elif mutation == "empty_journal":
        budget = Budget(tmp_path / "empty", 1)
    elif mutation == "changed_cost":
        item["attempts"][0]["cost_usd"] = 0
    elif mutation == "completed":
        (tmp_path / "results.json").write_text(json.dumps({"status": "complete"}))
    text = json.dumps(item) + "\n"
    (tmp_path / "responses.jsonl").write_text(text * (2 if mutation == "duplicate" else 1))
    with pytest.raises(ResumeError):
        load_checkpoint(tmp_path, splits, budget)


def test_checkpoint_can_continue_after_one_half_of_pair(tmp_path):
    budget, splits, item = checkpoint(tmp_path)
    measurements = load_checkpoint(tmp_path, splits, budget)
    assert measurements == [item]
    assert budget.charged == 0.001


def test_rejected_resume_does_not_relabel_completed_report(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    original = '{"status": "complete", "evidence": "preserve"}\n'
    (out / "results.json").write_text(original)
    journal = tmp_path / "budget.jsonl"
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"journal": str(journal.resolve())}))

    async def fail_before_validation(*args, **kwargs):
        raise ValueError("missing local configuration")

    monkeypatch.setattr(live_study, "study", fail_before_validation)
    monkeypatch.setattr(live_study, "validate_pilot_report", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_study",
            "--execute",
            "--mode",
            "full",
            "--resume",
            "--confirm-account-pricing",
            "--pilot-report",
            str(pilot),
            "--out",
            str(out),
            "--journal",
            str(journal),
        ],
    )
    with pytest.raises(SystemExit):
        live_study.main()
    assert (out / "results.json").read_text() == original
    assert not list(out.glob("results-before-resume-*"))


def test_results_replacement_preserves_previous_evidence(tmp_path):
    first, second = {"status": "incomplete"}, {"status": "complete"}
    live_study.save_run_result(tmp_path, first)
    live_study.save_run_result(tmp_path, second)
    assert json.loads((tmp_path / "results.json").read_bytes()) == second
    backups = list(tmp_path.glob("results-before-resume-*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_bytes()) == first
