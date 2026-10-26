"""Finite-sample CRC for bounded regret and the strict score > lambda policy."""

from bisect import bisect_right
from collections.abc import Sequence
from math import fsum, isfinite


def crc_lambda(
    scores: Sequence[float], regrets: Sequence[float], eps: float, B: float = 1.0
) -> float:
    """Return the smallest feasible threshold, or the zero-loss threshold 1.

    Corrected risk = (sum(regrets with score > lambda) + B)/(n+1).
    Binary search over unique scores handles ties together (right continuity).
    Empty/invalid data is an error, not a fabricated calibration artifact.
    """
    if not isfinite(B) or B <= 0 or not isfinite(eps) or not 0 <= eps <= B:
        raise ValueError("Require finite B > 0 and epsilon in [0, B].")
    if len(scores) == 0 or len(scores) != len(regrets):
        raise ValueError("Scores and regrets must have equal, nonzero length.")
    if any(not isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Scores must be finite and in [0, 1].")
    if any(not isfinite(r) or not 0 <= r <= B for r in regrets):
        raise ValueError("Regrets must be finite and in [0, B].")
    pairs = sorted(zip(scores, regrets, strict=True))
    ordered_scores = [s for s, _ in pairs]
    ordered_regrets = [r for _, r in pairs]
    candidates = sorted({0.0, 1.0, *scores})
    lo, hi = 0, len(candidates)
    while lo < hi:
        mid = (lo + hi) // 2
        start = bisect_right(ordered_scores, candidates[mid])
        corrected_risk = (fsum(ordered_regrets[start:]) + B) / (len(scores) + 1)
        if corrected_risk <= eps:
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo]) if lo < len(candidates) else 1.0
