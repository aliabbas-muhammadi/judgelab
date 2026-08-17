"""Tests for the provider interface and the deterministic fake provider."""

import pytest
from pydantic import ValidationError

from judgelab.providers import FakeProvider, JudgeProvider, JudgeRequest
from judgelab.types import (
    CandidateResponse,
    Example,
    Judge,
    JudgeVersion,
    Protocol,
    TaskType,
)


def _pairwise_request(
    example_id: str = "ex-1", order: tuple[str, ...] = ("a", "b")
) -> JudgeRequest:
    example = Example(
        id=example_id,
        task_type=TaskType.PAIRWISE,
        prompt="Which answer is better?",
        candidates=(CandidateResponse(id="a", text="A"), CandidateResponse(id="b", text="B")),
    )
    judge = Judge(
        id="fake-judge",
        provider="fake",
        protocol=Protocol.PAIRWISE,
        version=JudgeVersion(model="m", prompt_template_version="v1"),
    )
    return JudgeRequest(example=example, judge=judge, order=order)


def _pointwise_request(example_id: str = "ex-1") -> JudgeRequest:
    example = Example(
        id=example_id,
        task_type=TaskType.POINTWISE,
        prompt="Rate this answer.",
        candidates=(CandidateResponse(id="only", text="an answer"),),
    )
    judge = Judge(
        id="fake-judge",
        provider="fake",
        protocol=Protocol.POINTWISE,
        version=JudgeVersion(model="m", prompt_template_version="v1"),
    )
    return JudgeRequest(example=example, judge=judge, order=("only",))


def test_is_a_judge_provider() -> None:
    assert isinstance(FakeProvider(), JudgeProvider)


def test_deterministic_across_instances() -> None:
    req = _pairwise_request()
    assert FakeProvider().judge(req) == FakeProvider().judge(req)


def test_pairwise_output_shape() -> None:
    out = FakeProvider().judge(_pairwise_request())
    assert out.pairwise_choice is not None
    assert out.pointwise_score is None
    assert 0.0 <= (out.confidence or 0.0) <= 1.0


def test_pointwise_output_shape() -> None:
    out = FakeProvider().judge(_pointwise_request())
    assert out.pointwise_score is not None
    assert 0.0 <= out.pointwise_score <= 1.0
    assert out.pairwise_choice is None


def test_salt_changes_some_verdicts() -> None:
    plain = FakeProvider()
    salted = FakeProvider(salt="other-judge")
    differing = sum(
        plain.judge(_pairwise_request(f"ex-{i}")) != salted.judge(_pairwise_request(f"ex-{i}"))
        for i in range(20)
    )
    assert differing > 0


def test_order_swap_is_handled() -> None:
    # Both orderings are valid permutations and produce a (deterministic) verdict.
    ab = FakeProvider().judge(_pairwise_request(order=("a", "b")))
    ba = FakeProvider().judge(_pairwise_request(order=("b", "a")))
    assert ab.pairwise_choice is not None
    assert ba.pairwise_choice is not None


@pytest.mark.parametrize("bad_order", [("a", "c"), ("a",), ("a", "a"), ("a", "b", "b")])
def test_request_order_must_be_permutation(bad_order: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="permutation"):
        _pairwise_request(order=bad_order)
