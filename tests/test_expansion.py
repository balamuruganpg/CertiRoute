import json
import math
import pickle
from pathlib import Path

import pytest

from certiroute.monitor.drift import score_shift
from eval.benchmarks import answer_letter, extract_python, portfolio_metrics
from eval.drift import run as run_drift
from eval.import_benchmarks import Symbol, SymbolicUnpickler, fetch
from eval.multibench import run_task


def test_scoring_does_not_guess_letters_inside_prose():
    assert answer_letter("B") == "B"
    assert answer_letter("The answer is (C).") == "C"
    assert answer_letter("A or B") is None
    assert answer_letter("Because this is uncertain") is None
    assert extract_python("Here:\n```python\ndef f():\n    return 1\n```") == (
        "def f():\n    return 1\n"
    )


def test_combined_cost_is_weighted_before_savings():
    result = portfolio_metrics(
        [
            {"quality": 0.8, "regret": 0.02, "cost_usd": 2, "strong_cost_usd": 10},
            {"quality": 0.6, "regret": 0.04, "cost_usd": 1, "strong_cost_usd": 2},
        ]
    )
    assert result["quality"] == pytest.approx(0.7)
    assert result["regret"] == pytest.approx(0.03)
    assert result["savings_vs_strong"] == pytest.approx(0.75)


def test_drift_finite_horizon_and_ties():
    same = score_shift([0.1] * 100, [0.1] * 100, alpha=0.05, windows=10)
    shifted = score_shift([0.1] * 100, [0.9] * 100, alpha=0.05, windows=10)
    assert same["statistic"] == 0 and not same["detected"]
    assert shifted["statistic"] == 1 and shifted["detected"]
    for values in ([], [math.nan], [1.1]):
        with pytest.raises(ValueError):
            score_shift(values, [0.5], alpha=0.05, windows=10)


def test_pickle_reducers_are_inert_not_executed():
    import io

    class Payload:
        def __reduce__(self):
            return (eval, ("1/0",))

    result = SymbolicUnpickler(io.BytesIO(pickle.dumps(Payload()))).load()
    assert isinstance(result, Symbol)
    assert result.args == ("1/0",)


def test_import_rejects_modified_cache_before_decoding(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"corrupt")
    lock = json.loads(Path("data/eval/multibench/sources.lock.json").read_text())
    with pytest.raises(ValueError, match="checksum"):
        fetch(next(iter(lock)), path)


def test_coding_study_is_reproducible_and_splits_do_not_overlap(tmp_path):
    args = ("mbpp", Path("data/eval/multibench"))
    first = run_task(*args, tmp_path / "first", 3, 42)
    second = run_task(*args, tmp_path / "second", 3, 42)
    assert first["summaries"] == second["summaries"]
    assert first["router_sha256"] == second["router_sha256"]
    splits = json.loads((tmp_path / "first/split-manifest.json").read_text())
    groups = [splits["train_ids"], splits["first_calibration_ids"], splits["first_test_ids"]]
    flattened = [i for group in groups for i in group]
    assert len(flattened) == len(set(flattened))
    assert len(first["draws"]) == 3
    assert all(
        p["regret"] == 0 for p in first["draws"][0]["policies"] if p["policy"] == "strong-only"
    )


def test_drift_uses_disjoint_detection_audit_and_validation(tmp_path):
    result = run_drift(Path("results/gsm8k-shift"), tmp_path / "drift", repeats=5)
    assert len(result["summaries"]) == 2
    for path in (tmp_path / "drift").glob("*-splits.json"):
        split = json.loads(path.read_text())
        ids = [i for group in split.values() for i in group]
        assert len(ids) == len(set(ids)) == 600
