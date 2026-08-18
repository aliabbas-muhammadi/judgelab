"""GPT-4-judge vs human agreement on MT-Bench (reproduces & extends Zheng et al. 2023).

Reproduces the paper's judge-vs-human agreement and adds what it omitted:
chance-corrected Cohen's kappa with bootstrap confidence intervals, reported next
to raw agreement so the raw-vs-kappa gap is explicit.

Alignment (verified against the data): each verdict is reduced to a
model-identity preference in a canonical (alphabetical) ordering of the two
models, so human and GPT-4 verdicts align even when their row orderings differ.
The multiple human annotators per comparison are combined by majority vote
(vote ties -> TIE). The GPT-4 winner value ``tie (inconsistent)`` — a verdict
that flipped between the two presentation orderings — maps to TIE for agreement,
and is reported separately as the position-inconsistency rate.

Two setups follow the paper: S1 keeps ties; S2 excludes them (either side a tie).
"""

from __future__ import annotations

import enum
import math
from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict

from judgelab.datasets.mtbench import MtbenchSnapshot, MtbenchVote, MtbenchWinner
from judgelab.stats import bootstrap_ci, cohen_kappa, raw_agreement

_ROUND = 4


class Verdict(enum.StrEnum):
    """A verdict in the canonical model ordering."""

    FIRST = "first"
    SECOND = "second"
    TIE = "tie"


type ComparisonKey = tuple[int, tuple[str, str], int]


def _key(vote: MtbenchVote) -> ComparisonKey:
    pair = tuple(sorted((vote.model_a, vote.model_b)))
    return (vote.question_id, (pair[0], pair[1]), vote.turn)


def _verdict(vote: MtbenchVote) -> Verdict:
    if vote.winner is MtbenchWinner.MODEL_A:
        winner = vote.model_a
    elif vote.winner is MtbenchWinner.MODEL_B:
        winner = vote.model_b
    else:  # TIE or TIE_INCONSISTENT
        return Verdict.TIE
    canonical_first = min(vote.model_a, vote.model_b)
    return Verdict.FIRST if winner == canonical_first else Verdict.SECOND


def _majority(verdicts: list[Verdict]) -> Verdict:
    ranked = Counter(verdicts).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return Verdict.TIE  # no clear human majority
    return ranked[0][0]


def _round(value: float) -> float:
    return round(value, _ROUND)


class Metric(BaseModel):
    """Agreement on a set of aligned comparisons."""

    model_config = ConfigDict(frozen=True)

    n: int
    raw_agreement: float
    cohen_kappa: float
    kappa_ci_low: float
    kappa_ci_high: float


class AgreementResult(BaseModel):
    """The full MT-Bench GPT-4-vs-human agreement result."""

    model_config = ConfigDict(frozen=True)

    n_comparisons: int
    n_human_votes: int
    n_gpt4_verdicts: int
    position_inconsistency_rate: float
    with_ties: Metric
    ties_excluded: Metric
    by_turn: dict[int, Metric]


def _metric(
    human: list[Verdict],
    gpt4: list[Verdict],
    *,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> Metric:
    n = len(human)
    raw = raw_agreement(human, gpt4)
    kappa = cohen_kappa(human, gpt4)
    if n >= 2 and not math.isnan(kappa):
        low, high = bootstrap_ci(
            cohen_kappa, human, gpt4, confidence=confidence, n_resamples=n_resamples, seed=seed
        )
    else:
        low = high = math.nan
    return Metric(
        n=n,
        raw_agreement=_round(raw),
        cohen_kappa=_round(kappa),
        kappa_ci_low=_round(low),
        kappa_ci_high=_round(high),
    )


def compute_agreement(
    snapshot: MtbenchSnapshot,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> AgreementResult:
    """Align the two splits by comparison and compute agreement (S1, S2, per-turn)."""
    human_by_key: dict[ComparisonKey, list[Verdict]] = defaultdict(list)
    for vote in snapshot.human:
        human_by_key[_key(vote)].append(_verdict(vote))

    gpt4_by_key: dict[ComparisonKey, Verdict] = {}
    gpt4_inconsistent: dict[ComparisonKey, bool] = {}
    for vote in snapshot.gpt4_pair:
        key = _key(vote)
        gpt4_by_key[key] = _verdict(vote)
        gpt4_inconsistent[key] = vote.winner is MtbenchWinner.TIE_INCONSISTENT

    keys = sorted(key for key in human_by_key if key in gpt4_by_key)
    human: list[Verdict] = [_majority(human_by_key[key]) for key in keys]
    gpt4: list[Verdict] = [gpt4_by_key[key] for key in keys]
    turns: list[int] = [key[2] for key in keys]
    inconsistency = [gpt4_inconsistent[key] for key in keys]

    decisive = [
        (h, g)
        for h, g in zip(human, gpt4, strict=True)
        if h is not Verdict.TIE and g is not Verdict.TIE
    ]
    by_turn: dict[int, Metric] = {}
    for turn in sorted(set(turns)):
        th = [h for h, t in zip(human, turns, strict=True) if t == turn]
        tg = [g for g, t in zip(gpt4, turns, strict=True) if t == turn]
        by_turn[turn] = _metric(th, tg, n_resamples=n_resamples, seed=seed, confidence=confidence)

    return AgreementResult(
        n_comparisons=len(keys),
        n_human_votes=len(snapshot.human),
        n_gpt4_verdicts=len(snapshot.gpt4_pair),
        position_inconsistency_rate=_round(sum(inconsistency) / len(inconsistency)),
        with_ties=_metric(human, gpt4, n_resamples=n_resamples, seed=seed, confidence=confidence),
        ties_excluded=_metric(
            [h for h, _ in decisive],
            [g for _, g in decisive],
            n_resamples=n_resamples,
            seed=seed,
            confidence=confidence,
        ),
        by_turn=by_turn,
    )


__all__ = ["AgreementResult", "Metric", "Verdict", "compute_agreement"]
