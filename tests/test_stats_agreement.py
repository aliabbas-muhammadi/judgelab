"""Correctness tests for the agreement statistics.

The math is pinned two ways: hand-computed kappa on known confusion tables, and
an independent cross-check against scikit-learn over random and correlated data.
"""

import math
import random

import pytest
from sklearn.metrics import cohen_kappa_score

from judgelab.stats import cohen_kappa, raw_agreement


def _from_confusion(c00: int, c01: int, c10: int, c11: int) -> tuple[list[int], list[int]]:
    """Build paired label lists realising a 2x2 confusion table [[c00,c01],[c10,c11]]."""
    a = [0] * c00 + [0] * c01 + [1] * c10 + [1] * c11
    b = [0] * c00 + [1] * c01 + [0] * c10 + [1] * c11
    return a, b


def test_raw_agreement_basic() -> None:
    assert raw_agreement(["a", "b", "b"], ["a", "b", "x"]) == pytest.approx(2 / 3)


def test_kappa_hand_computed_table() -> None:
    # [[20,5],[10,15]]: p_o=0.70, p_e=0.50 -> kappa=0.40 (textbook value).
    a, b = _from_confusion(20, 5, 10, 15)
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_perfect_agreement() -> None:
    a = ["A", "B", "tie", "A", "B"]
    assert cohen_kappa(a, list(a)) == pytest.approx(1.0)
    assert raw_agreement(a, list(a)) == 1.0


def test_kappa_zero_when_independent() -> None:
    # [[1,1],[1,1]]: p_o=0.5, p_e=0.5 -> kappa=0.
    a, b = _from_confusion(1, 1, 1, 1)
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def test_kappa_nan_without_class_variance() -> None:
    a = ["x", "x", "x", "x"]
    assert math.isnan(cohen_kappa(a, list(a)))
    assert raw_agreement(a, list(a)) == 1.0  # raw agreement is still reported


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        cohen_kappa([1, 2], [1])


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        cohen_kappa([], [])


def test_matches_sklearn_on_random_labels() -> None:
    labels = ["A", "B", "tie"]
    for seed in range(10):
        rng = random.Random(seed)
        a = [rng.choice(labels) for _ in range(200)]
        b = [rng.choice(labels) for _ in range(200)]
        assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-9)


def test_matches_sklearn_on_correlated_labels() -> None:
    labels = ["A", "B", "tie"]
    for seed in range(10):
        rng = random.Random(1000 + seed)
        a = [rng.choice(labels) for _ in range(200)]
        b = [x if rng.random() < 0.8 else rng.choice(labels) for x in a]  # ~80% agreement
        assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-9)
