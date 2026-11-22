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


# Two-sided 95% quantiles of Student's t for 1..30 degrees of freedom; above 30 the normal
# quantile is within 2% and is used instead.
_T_975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip
EXACT_SIGN_FLIP_MAX = 20
SIGN_FLIP_DRAWS = 100_000


def t_quantile_975(df: int) -> float:
    """Return the 0.975 quantile of Student's t with `df` degrees of freedom."""
    if df < 1:
        msg = "degrees of freedom must be positive"
        raise ValueError(msg)
    return _T_975[df - 1] if df <= len(_T_975) else Z_95


def paired_t_interval(diffs: Sequence[float]) -> tuple[float, float, float] | None:
    """Mean of paired differences with its 95% t-interval (df = n - 1).

    Args:
        diffs: One difference per pair, for example per scenario.

    Returns:
        `(mean, low, high)`, or `None` with fewer than two differences.
    """
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half = t_quantile_975(n - 1) * math.sqrt(variance / n)
    return mean, mean - half, mean + half


def sign_flip_p(
    diffs: Sequence[float], *, alternative: str = "two-sided", seed: int = 0
) -> float | None:
    """P-value of the sign-flip permutation test that the paired differences are centred on 0.

    Under the null every difference is equally likely to have either sign, so the test compares
    the observed mean with the means of all 2^n sign assignments (exact up to
    `EXACT_SIGN_FLIP_MAX` pairs, else `SIGN_FLIP_DRAWS` seeded random assignments).

    Args:
        diffs: Paired differences.
        alternative: `two-sided`, `greater` (mean > 0) or `less` (mean < 0).
        seed: Seed of the random assignments for large n.

    Returns:
        The p-value, or `None` without differences.
    """
    n = len(diffs)
    if n == 0:
        return None
    observed = sum(diffs)
    eps = 1e-12 * (1 + sum(abs(d) for d in diffs))

    def extreme(total: float) -> bool:
        if alternative == "greater":
            return total >= observed - eps
        if alternative == "less":
            return total <= observed + eps
        return abs(total) >= abs(observed) - eps

    if n <= EXACT_SIGN_FLIP_MAX:
        hits = 0
        for mask in range(1 << n):
            total = sum(-d if mask >> i & 1 else d for i, d in enumerate(diffs))
            hits += extreme(total)
        return hits / (1 << n)
    rng = random.Random(seed)
    hits = sum(
        extreme(sum(d if rng.random() < 0.5 else -d for d in diffs))  # noqa: S311
        for _ in range(SIGN_FLIP_DRAWS)
    )
    return (hits + 1) / (SIGN_FLIP_DRAWS + 1)


def bca_interval[T](
    clusters: Sequence[T],
    statistic: Callable[[Sequence[T]], float | None],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float] | None:
    """95% bias-corrected and accelerated bootstrap interval over resampled clusters.

    Resamples where the statistic is undefined (`None`) are dropped.

    Args:
        clusters: Independent units, for example scenarios.
        statistic: Function of a resampled multiset of clusters.
        resamples: Number of resamples.
        seed: Seed of the resampling.

    Returns:
        `(low, high)`, or `None` when the statistic is undefined on the data, there are fewer
        than two clusters, or too few resamples are defined.
    """
    estimate = statistic(clusters)
    if estimate is None or len(clusters) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        value = statistic(rng.choices(clusters, k=len(clusters)))
        if value is not None:
            draws.append(value)
    if len(draws) < resamples // 2:
        return None
    draws.sort()
    below = sum(v < estimate for v in draws) + 0.5 * sum(v == estimate for v in draws)
    proportion = min(max(below / len(draws), 1 / len(draws)), 1 - 1 / len(draws))
    z0 = NormalDist().inv_cdf(proportion)
    jack = [statistic([c for j, c in enumerate(clusters) if j != i]) for i in range(len(clusters))]
    defined = [v for v in jack if v is not None]
    acceleration = 0.0
    if len(defined) == len(clusters):
        centre = sum(defined) / len(defined)
        num = sum((centre - v) ** 3 for v in defined)
        den = 6 * sum((centre - v) ** 2 for v in defined) ** 1.5
        acceleration = num / den if den else 0.0
    bounds = []
    for tail in (_TAIL, 1 - _TAIL):
        z = NormalDist().inv_cdf(tail)
        adjusted = NormalDist().cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
        bounds.append(quantile(draws, adjusted))
    return bounds[0], bounds[1]


def holm(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-adjusted p-values (family-wise error), in the input order; `None` stays `None`."""
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    m = len(indexed)
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted
