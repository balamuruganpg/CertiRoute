"""Bounded regret relative to the strong model."""

from math import isfinite


def regret(q_strong: float, q_cheap: float) -> float:
    if any(not isfinite(q) or not 0 <= q <= 1 for q in (q_strong, q_cheap)):
        raise ValueError("Quality scores must be finite and in [0, 1].")
    return max(0.0, q_strong - q_cheap)
