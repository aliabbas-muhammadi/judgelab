"""Tests for the MT-Bench GPT-4-vs-human agreement computation."""

import pytest

from judgelab.benchmarks import compute_agreement
from judgelab.datasets.mtbench import MtbenchSnapshot, MtbenchVote, MtbenchWinner, load_snapshot

W = MtbenchWinner


def _vote(
    q: int, a: str, b: str, winner: MtbenchWinner, judge: str = "j", turn: int = 1
) -> MtbenchVote:
    return MtbenchVote(question_id=q, model_a=a, model_b=b, winner=winner, judge=judge, turn=turn)


def _snap(human: tuple[MtbenchVote, ...], gpt4: tuple[MtbenchVote, ...]) -> MtbenchSnapshot:
    return MtbenchSnapshot(human=human, gpt4_pair=gpt4)


def test_order_invariant_alignment() -> None:
    # Human sees (x,y) and prefers x; GPT-4 sees the SWAPPED (y,x) and also prefers x.
    # Same model identity -> agreement, despite different row orderings.
    human = (_vote(1, "x", "y", W.MODEL_A),)
    gpt4 = (_vote(1, "y", "x", W.MODEL_B),)
    result = compute_agreement(_snap(human, gpt4), n_resamples=100)
    assert result.n_comparisons == 1
    assert result.with_ties.raw_agreement == 1.0


def test_hand_computed_raw_and_kappa() -> None:
    # 4 comparisons; raw=3/4=0.75; kappa=0.4375/0.6875=0.6364 (hand-computed).
    human = (
        _vote(1, "a", "b", W.MODEL_A),
        _vote(2, "a", "b", W.MODEL_A),
        _vote(3, "a", "b", W.MODEL_B),
        _vote(4, "a", "b", W.TIE),
    )
    gpt4 = (
        _vote(1, "a", "b", W.MODEL_A),  # agree
        _vote(2, "a", "b", W.MODEL_B),  # disagree
        _vote(3, "a", "b", W.MODEL_B),  # agree
        _vote(4, "a", "b", W.TIE),  # agree
    )
    result = compute_agreement(_snap(human, gpt4), n_resamples=100)
    assert result.with_ties.n == 4
    assert result.with_ties.raw_agreement == 0.75
    assert result.with_ties.cohen_kappa == pytest.approx(0.6364, abs=1e-4)


def test_majority_vote_over_annotators() -> None:
    # Comparison 1: three human votes FIRST,FIRST,SECOND -> majority FIRST.
    # Comparison 2: two human votes FIRST,SECOND -> a vote tie -> TIE.
    human = (
        _vote(1, "a", "b", W.MODEL_A),
        _vote(1, "a", "b", W.MODEL_A),
        _vote(1, "a", "b", W.MODEL_B),
        _vote(2, "a", "b", W.MODEL_A),
        _vote(2, "a", "b", W.MODEL_B),
    )
    gpt4 = (_vote(1, "a", "b", W.MODEL_A), _vote(2, "a", "b", W.TIE))
    result = compute_agreement(_snap(human, gpt4), n_resamples=50)
    assert result.n_comparisons == 2
    assert result.n_human_votes == 5
    assert result.with_ties.raw_agreement == 1.0  # majority FIRST vs FIRST; vote-tie TIE vs TIE


def test_position_inconsistency_counted_and_treated_as_tie() -> None:
    human = (_vote(1, "a", "b", W.MODEL_A), _vote(2, "a", "b", W.MODEL_A))
    gpt4 = (_vote(1, "a", "b", W.TIE_INCONSISTENT), _vote(2, "a", "b", W.MODEL_A))
    result = compute_agreement(_snap(human, gpt4), n_resamples=50)
    assert result.position_inconsistency_rate == 0.5
    # C1: human FIRST vs gpt4 TIE (from inconsistent) -> disagree; C2 agree -> raw 0.5
    assert result.with_ties.raw_agreement == 0.5


def test_ties_excluded_setup() -> None:
    human = (_vote(1, "a", "b", W.MODEL_A), _vote(2, "a", "b", W.TIE))
    gpt4 = (_vote(1, "a", "b", W.MODEL_A), _vote(2, "a", "b", W.MODEL_B))
    result = compute_agreement(_snap(human, gpt4), n_resamples=50)
    assert result.with_ties.n == 2
    assert result.ties_excluded.n == 1  # the TIE comparison is dropped


def test_deterministic_with_seed() -> None:
    snapshot = load_snapshot()
    first = compute_agreement(snapshot, n_resamples=200, seed=0)
    second = compute_agreement(snapshot, n_resamples=200, seed=0)
    assert first == second


def test_real_snapshot_structural_sanity() -> None:
    result = compute_agreement(load_snapshot(), n_resamples=200, seed=0)
    assert result.n_comparisons == 1814
    assert result.n_human_votes == 3355
    assert result.n_gpt4_verdicts == 2400
    assert 0.0 <= result.position_inconsistency_rate <= 1.0
    # Decisive (ties-excluded) comparisons should agree more than the with-ties set.
    assert result.ties_excluded.raw_agreement > result.with_ties.raw_agreement
    assert set(result.by_turn) == {1, 2}
    for metric in (result.with_ties, result.ties_excluded, *result.by_turn.values()):
        assert 0.0 <= metric.raw_agreement <= 1.0
        assert -1.0 <= metric.cohen_kappa <= 1.0
