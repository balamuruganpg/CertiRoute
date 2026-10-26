import math
import random

import pytest

from certiroute.calibration.crc import crc_lambda
from certiroute.calibration.losses import regret


def test_crc_ties_endpoints_and_empty_feasible_set():
    assert crc_lambda([0.2, 0.8, 0.8], [1, 1, 1], 0.5) == 0.8
    assert crc_lambda([0, 1], [1, 1], 0.5) == 1
    assert crc_lambda([0, 0], [1, 1], 1 / 3) == 0
    assert crc_lambda([0.9] * 5, [0] * 5, 0.01) == 1
    assert crc_lambda([0.9] * 99, [0] * 99, 0.01) == 0
    assert crc_lambda([0.9] * 99, [0] * 99, 0) == 1


def test_crc_matches_brute_force_and_epsilon_monotonicity():
    rng = random.Random(42)
    for _ in range(100):
        scores = [rng.choice([0, 0.2, 0.5, 0.8, 1]) for _ in range(30)]
        regrets = [rng.random() for _ in scores]
        thresholds = []
        for epsilon in (0.01, 0.05, 0.1, 0.2, 0.5):
            candidates = sorted({0.0, 1.0, *scores})
            feasible = [
                lam
                for lam in candidates
                if (math.fsum(r for s, r in zip(scores, regrets, strict=True) if s > lam) + 1)
                / (len(scores) + 1)
                <= epsilon
            ]
            actual = crc_lambda(scores, regrets, epsilon)
            assert actual == (min(feasible) if feasible else 1)
            thresholds.append(actual)
        assert thresholds == sorted(thresholds, reverse=True)


@pytest.mark.parametrize(
    "scores,regrets,epsilon",
    [
        ([], [], 0.1),
        ([0.2], [], 0.1),
        ([float("nan")], [0], 0.1),
        ([1.1], [0], 0.1),
        ([0.5], [-1], 0.1),
        ([0.5], [float("inf")], 0.1),
        ([0.5], [0], float("nan")),
        ([0.5], [0], -0.1),
    ],
)
def test_crc_rejects_invalid_inputs(scores, regrets, epsilon):
    with pytest.raises(ValueError):
        crc_lambda(scores, regrets, epsilon)


def test_regret_bounds():
    assert regret(0.2, 0.8) == 0
    assert regret(1, 0) == 1
    with pytest.raises(ValueError):
        regret(float("nan"), 0)
