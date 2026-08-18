"""Percentile bootstrap confidence intervals.

Resamples one or more *paired* samples with replacement using the same indices
for every sample, so paired statistics (e.g. judge-vs-human agreement) resample
correctly. Seeded, so a committed report's intervals recompute identically under
the keyless drift gate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    statistic: Callable[..., float],
    *samples: Sequence[object],
    confidence: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over one or more paired samples.

    ``statistic`` receives the resampled samples (in the same order) and returns a
    scalar. Returns the ``(low, high)`` percentile interval at ``confidence``.
    """
    if not samples:
        raise ValueError("need at least one sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    n = len(samples[0])
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if any(len(sample) != n for sample in samples):
        raise ValueError("all samples must have the same length")

    arrays = [np.asarray(sample) for sample in samples]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        index = rng.integers(0, n, size=n)
        estimates[i] = statistic(*(array[index] for array in arrays))

    alpha = 1.0 - confidence
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


__all__ = ["bootstrap_ci"]
