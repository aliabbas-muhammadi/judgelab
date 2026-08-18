"""Chance-corrected agreement statistics.

Raw agreement overstates reliability — the MT-Bench reproduction shows a large
gap between raw agreement and Cohen's kappa — so judge-vs-human and
judge-vs-judge comparisons always report kappa alongside raw agreement and n.

Kappa is undefined when a rater shows no class variance (both raters unanimous
on a single class gives chance agreement 1 and a 0/0 form); that case returns
NaN by design, and callers should surface raw agreement and n rather than a
bare kappa.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence


def _paired_len(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    if len(a) != len(b):
        raise ValueError(f"paired sequences differ in length: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("need at least one paired observation")
    return len(a)


def raw_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Fraction of paired observations on which the two raters agree."""
    n = _paired_len(a, b)
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    return agree / n


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's (1960) kappa for two raters on nominal labels.

    Returns NaN when expected (chance) agreement is 1 — i.e. neither rater varies
    — since kappa is then a 0/0 form and undefined.
    """
    n = _paired_len(a, b)
    p_o = raw_agreement(a, b)
    count_a = Counter(a)
    count_b = Counter(b)
    labels = count_a.keys() | count_b.keys()
    p_e = sum((count_a.get(label, 0) / n) * (count_b.get(label, 0) / n) for label in labels)
    if math.isclose(p_e, 1.0):
        return math.nan
    return (p_o - p_e) / (1.0 - p_e)


__all__ = ["cohen_kappa", "raw_agreement"]
