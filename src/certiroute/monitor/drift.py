"""Finite-horizon, two-sample DKW score-distribution alert (not a risk certificate)."""

import math
from bisect import bisect_right


def score_shift(
    reference: list[float], current: list[float], *, alpha: float = 0.05, windows: int = 10
) -> dict[str, float | bool]:
    """Union-bound two DKW bands and allocate alpha across predeclared windows.

    IID observations within each sample are required. Reusing a reference across
    windows is allowed by the union bound. Autocorrelated production traffic does
    not inherit this bound. A score-only detector can miss conditional label drift.
    """
    if not reference or not current or not 0 < alpha < 1 or windows < 1:
        raise ValueError("Nonempty samples and a valid finite horizon are required.")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in reference + current):
        raise ValueError("Scores must be finite and in [0,1].")
    a, b = sorted(reference), sorted(current)
    distance = max(
        abs(bisect_right(a, x) / len(a) - bisect_right(b, x) / len(b)) for x in set(a + b)
    )
    log_term = math.log(4 * windows / alpha)
    threshold = math.sqrt(log_term / (2 * len(a))) + math.sqrt(log_term / (2 * len(b)))
    return {"statistic": distance, "threshold": threshold, "detected": distance > threshold}
