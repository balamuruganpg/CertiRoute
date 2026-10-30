import json

import pytest

from eval.publish_live import export_snapshot, verify_snapshot


def test_export_excludes_private_metadata_and_preserves_metrics(tmp_path):
    source = tmp_path / "private"
    source.mkdir()
    observation = {
        "id": "case-1",
        "phase": "test",
        "policy": "CertiRoute",
        "status": 200,
        "ok": True,
        "output": "7",
        "quality": 1.0,
        "regret": 0.0,
        "latency_ms": 12,
        "cost_usd": 0.01,
        "cost_known": True,
        "attempts": [
            {
                "id": "PRIVATE-ID",
                "upstream_id": "PRIVATE-ID",
                "tier": "cheap",
                "ok": True,
                "cost_usd": 0.01,
                "cost_known": True,
                "latency_ms": 12,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 8, "private": "PRIVATE-KEY"},
                },
            }
        ],
    }
    report = {
        "status": "complete",
        "spec": {},
        "execution": {"journal": "PRIVATE-PATH"},
        "observations": [observation],
        "summaries": {},
        "api_key": "PRIVATE-KEY",
    }
    (source / "results.json").write_text(json.dumps(report))
    (source / "splits.json").write_text(
        json.dumps({"train": [], "calibration": [], "test": ["case-1"]})
    )
    out = tmp_path / "public"
    export_snapshot(source, out)
    all_text = "".join(p.read_text() for p in out.iterdir())
    assert "PRIVATE-" not in all_text
    saved = json.loads((out / "observations.jsonl").read_text())
    assert saved["output"] == "7" and saved["quality"] == 1
    assert saved["cost_usd"] == observation["cost_usd"]
    assert saved["attempts"][0]["usage"]["prompt_tokens_details"] == {"cached_tokens": 8}
    with pytest.raises(FileExistsError):
        export_snapshot(source, out)


def test_incomplete_study_cannot_be_published(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "results.json").write_text('{"status": "incomplete"}')
    with pytest.raises(ValueError, match="complete"):
        export_snapshot(source, tmp_path / "out")


def test_published_live_metrics_recompute_offline():
    from pathlib import Path

    verify_snapshot(Path("results/live-gsm8k-v1"))


@pytest.mark.parametrize("field", ["regret", "test_id", "cached_cost"])
def test_verify_rejects_inconsistent_derived_evidence(tmp_path, field):
    import hashlib
    from pathlib import Path

    root = Path("results/live-gsm8k-v1")
    report = json.loads((root / "results.json").read_bytes())
    rows = [json.loads(line) for line in (root / "observations.jsonl").read_bytes().splitlines()]
    splits = json.loads((root / "splits.json").read_bytes())
    if field == "regret":
        row = next(
            r
            for r in rows
            if r["phase"] == "test" and r["policy"] == "CertiRoute" and r["regret"] == 1
        )
        row["regret"] = 0
        report["summaries"]["CertiRoute"]["regret"] -= 1 / 99
    elif field == "test_id":
        splits["test"][0] = "not-in-the-held-out-set"
    else:
        attempt = next(
            a
            for r in rows
            for a in r["attempts"]
            if a.get("cost_basis") == "actual_usage_times_recorded_cache_prices"
        )
        attempt["usage"]["prompt_tokens_details"]["cached_tokens"] += 1
    raw = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    (tmp_path / "observations.jsonl").write_bytes(raw)
    report["publication"]["observations_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "results.json").write_text(json.dumps(report))
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    with pytest.raises(ValueError):
        verify_snapshot(tmp_path)
