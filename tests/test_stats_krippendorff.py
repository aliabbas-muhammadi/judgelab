"""Tests for nominal Krippendorff's alpha."""

import math
import random

import krippendorff as krippendorff_pkg
import numpy as np
import pytest

from judgelab.stats import krippendorff_alpha


def test_perfect_agreement() -> None:
    # Each unit internally unanimous, units differ -> alpha = 1.
    assert krippendorff_alpha([[0, 0], [1, 1], [0, 0]]) == pytest.approx(1.0)


def test_systematic_within_unit_disagreement_is_negative() -> None:
    # units [[0,1],[0,1]]: n=4, diagonal=0, denom=8 -> alpha = 1 - 3*4/8 = -0.5 (hand-computed).
    assert krippendorff_alpha([[0, 1], [0, 1]]) == pytest.approx(-0.5)


def test_single_category_is_nan() -> None:
    assert math.isnan(krippendorff_alpha([[0, 0, 0]]))


def test_units_with_one_rater_are_ignored() -> None:
    # The singleton unit contributes nothing; result equals the paired unit alone.
    with_singleton = krippendorff_alpha([[0, 0], [1, 1], [2]])
    without = krippendorff_alpha([[0, 0], [1, 1]])
    assert with_singleton == pytest.approx(without)


def test_no_pairable_units_raises() -> None:
    with pytest.raises(ValueError, match="two or more raters"):
        krippendorff_alpha([[0], [1], [2]])


def test_matches_krippendorff_package() -> None:
    categories = [0, 1, 2]
    max_raters = 5
    for seed in range(10):
        rng = random.Random(seed)
        units: list[list[int]] = []
        padded: list[list[float]] = []
        for _ in range(60):
            m = rng.randint(2, max_raters)
            values = [rng.choice(categories) for _ in range(m)]
            units.append(values)
            padded.append([*values, *([math.nan] * (max_raters - m))])
        reliability = np.array(padded, dtype=float).T  # rows = raters, cols = units
        mine = krippendorff_alpha(units)
        theirs = krippendorff_pkg.alpha(
            reliability_data=reliability, level_of_measurement="nominal"
        )
        assert mine == pytest.approx(theirs, abs=1e-9)
