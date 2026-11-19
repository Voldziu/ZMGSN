"""Interval estimates and repeat statistics for the evaluation report (standard library only)."""

import math
import random
from collections.abc import Callable, Sequence
from statistics import NormalDist

Z_95 = NormalDist().inv_cdf(0.975)  # 1.959963984540054
BOOTSTRAP_RESAMPLES = 10_000
_TAIL = 0.025  # each tail of a 95% interval


def wilson_interval(successes: int, n: int) -> tuple[float, float] | None:
    """Return the 95% Wilson score interval of a binomial proportion.

    Args:
        successes: Number of successes.
        n: Number of trials.

    Returns:
        Lower and upper bound, exact at 0 and 1 when no or every trial succeeded; `None`
        without trials.

    Raises:
        ValueError: If `successes` is not between 0 and `n`.
    """
    if not 0 <= successes <= n:
        msg = f"successes must be between 0 and n={n}, got {successes}"
        raise ValueError(msg)
    if n == 0:
        return None
    p = successes / n
    z2 = Z_95 * Z_95
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == n else min(1.0, center + half)
    return lower, upper


def pass_hat_k(counts: Sequence[tuple[int, int]], k: int) -> float | None:
    """Return pass^k, the mean over scenarios of C(c, k) / C(n, k).

    With c successes in n repeats of a scenario, C(c, k) / C(n, k) is the unbiased estimate of
    the probability that k repeats of that scenario all succeed (as in tau-bench).

    Args:
        counts: `(successes, runs)` per scenario.
        k: Number of repeats that must all succeed (at least 1).

    Returns:
        pass^k, or `None` if there is no scenario or some scenario has fewer than `k` runs.

    Raises:
        ValueError: If `k` is below 1.
    """
    if k < 1:
        msg = f"k must be at least 1, got {k}"
        raise ValueError(msg)
    if not counts or any(n < k for _, n in counts):
        return None
    return sum(math.comb(c, k) / math.comb(n, k) for c, n in counts) / len(counts)


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Return the q-quantile of sorted values, interpolating linearly between ranks.

    Args:
        sorted_values: Values in ascending order.
        q: Quantile between 0 and 1.

    Returns:
        The quantile.

    Raises:
        ValueError: If there are no values or `q` is outside 0..1.
    """
    if not sorted_values or not 0 <= q <= 1:
        msg = f"need values and 0 <= q <= 1, got {len(sorted_values)} values and q={q}"
        raise ValueError(msg)
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def cluster_bootstrap[T](
    clusters: Sequence[T],
    statistics: Sequence[Callable[[Sequence[T]], float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Return 95% percentile intervals of statistics when whole clusters are resampled.

    Each resample draws `len(clusters)` clusters with replacement, and every statistic is
    computed on the same draws. The seed makes the intervals reproducible.

    Args:
        clusters: Independent units, for example the scenarios of a comparison.
        statistics: Functions of a resampled multiset of clusters.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        One `(lower, upper)` interval per statistic.

    Raises:
        ValueError: If there is no cluster.
    """
    if not clusters:
        msg = "cannot resample zero clusters"
        raise ValueError(msg)
    rng = random.Random(seed)
    draws: list[list[float]] = [[] for _ in statistics]
    for _ in range(resamples):
        sample = rng.choices(clusters, k=len(clusters))
        for values, statistic in zip(draws, statistics, strict=True):
            values.append(statistic(sample))
    intervals = []
    for values in draws:
        values.sort()
        intervals.append((quantile(values, _TAIL), quantile(values, 1 - _TAIL)))
    return intervals
