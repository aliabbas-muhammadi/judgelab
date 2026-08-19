"""Krippendorff's alpha (nominal) for inter-rater reliability.

Unlike Cohen's kappa (exactly two fixed raters), Krippendorff's alpha handles any
number of raters, units rated by different subsets of raters, and missing data —
which is exactly the shape of the MT-Bench human annotations (1-7 annotators per
comparison, different annotators per comparison). It is the right tool for the
human-human agreement ceiling that contextualises judge-human agreement.

Input is one sequence of rater values per unit; units with fewer than two values
contribute no pairable observations and are ignored. Only the nominal difference
metric is implemented (categorical verdicts).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence


def krippendorff_alpha(units: Iterable[Sequence[Hashable]]) -> float:
    """Nominal Krippendorff's alpha over per-unit rater values.

    Returns NaN when there is no possible disagreement (every pairable value is the
    same category), since alpha is then undefined.
    """
    diagonal = 0.0  # sum over units of within-unit same-category pairs, weighted by 1/(m-1)
    totals: Counter[Hashable] = Counter()  # category -> pairable value mass n_c
    for unit in units:
        values = list(unit)
        m = len(values)
        if m < 2:
            continue
        counts = Counter(values)
        for category, count in counts.items():
            diagonal += count * (count - 1) / (m - 1)
            totals[category] += count

    n = sum(totals.values())
    if n < 2:
        raise ValueError("need at least one unit rated by two or more raters")

    sum_squares = sum(count * count for count in totals.values())
    denominator = n * n - sum_squares
    if denominator == 0:  # all pairable values share one category -> no possible disagreement
        return math.nan
    return 1.0 - (n - 1) * (n - diagonal) / denominator


__all__ = ["krippendorff_alpha"]
