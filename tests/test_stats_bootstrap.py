"""Tests for percentile bootstrap confidence intervals."""

import random

import numpy as np
import pytest
from scipy.stats import bootstrap as scipy_bootstrap

from judgelab.stats import bootstrap_ci, cohen_kappa


def _mean(sample: object) -> float:
    return float(np.mean(sample))


def test_reproducible_with_seed() -> None:
    data = list(range(50))
    first = bootstrap_ci(_mean, data, seed=7, n_resamples=500)
    second = bootstrap_ci(_mean, data, seed=7, n_resamples=500)
    assert first == second


def test_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(5.0, 2.0, size=300)
    low, high = bootstrap_ci(_mean, data, confidence=0.95, n_resamples=2000, seed=0)
    assert low <= float(np.mean(data)) <= high


def test_matches_scipy_percentile_bootstrap() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 1.0, size=200)
    mine = bootstrap_ci(_mean, data, confidence=0.95, n_resamples=5000, seed=0)
    ref = scipy_bootstrap(
        (data,),
        np.mean,
        confidence_level=0.95,
        n_resamples=5000,
        method="percentile",
        random_state=0,
    ).confidence_interval
    assert mine[0] == pytest.approx(ref.low, abs=0.05)
    assert mine[1] == pytest.approx(ref.high, abs=0.05)


def test_paired_kappa_ci_brackets_point_estimate() -> None:
    labels = ["A", "B", "tie"]
    rng = random.Random(0)
    a = [rng.choice(labels) for _ in range(300)]
    b = [x if rng.random() < 0.75 else rng.choice(labels) for x in a]
    point = cohen_kappa(a, b)
    low, high = bootstrap_ci(cohen_kappa, a, b, confidence=0.95, n_resamples=1000, seed=0)
    assert low < high
    assert low <= point <= high


def test_higher_confidence_is_wider() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=200)
    lo90, hi90 = bootstrap_ci(_mean, data, confidence=0.90, n_resamples=2000, seed=0)
    lo99, hi99 = bootstrap_ci(_mean, data, confidence=0.99, n_resamples=2000, seed=0)
    assert (hi99 - lo99) > (hi90 - lo90)


def test_no_samples_raises() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        bootstrap_ci(_mean)


def test_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci(_mean, [])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci(cohen_kappa, [1, 2, 3], [1, 2])


def test_bad_confidence_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(_mean, [1, 2, 3], confidence=1.5)
