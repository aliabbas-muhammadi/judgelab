"""Validation and round-trip tests for the result-side domain models."""

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from judgelab.types import (
    EvaluationRun,
    ExperimentConfig,
    HumanLabel,
    Judge,
    JudgeOutput,
    JudgeVersion,
    PairwiseChoice,
    Protocol,
    RunStatus,
    Trial,
    Usage,
)

FIXED_TIME = datetime(2026, 8, 16, 12, 0, 0, tzinfo=UTC)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="run",
        dataset_id="mtbench",
        dataset_hash=hashlib.sha256(b"x").hexdigest(),
        judge=Judge(
            id="j",
            provider="fake",
            protocol=Protocol.PAIRWISE,
            version=JudgeVersion(model="m", prompt_template_version="v1"),
        ),
    )


def test_judge_output_pairwise_valid() -> None:
    out = JudgeOutput(pairwise_choice=PairwiseChoice.A, raw_text="A is better")
    assert out.pairwise_choice is PairwiseChoice.A


def test_judge_output_pointwise_valid() -> None:
    out = JudgeOutput(pointwise_score=0.8, usage=Usage(prompt_tokens=10, completion_tokens=2))
    assert out.pointwise_score == 0.8


def test_judge_output_failure_carries_no_verdict() -> None:
    out = JudgeOutput(parse_error=True, raw_text="garbage")
    assert out.parse_error is True
    assert out.pairwise_choice is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parse_error": True, "pairwise_choice": PairwiseChoice.A},
        {"error": "429 rate limit", "pointwise_score": 0.5},
    ],
)
def test_failed_output_with_verdict_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="failed judge output"):
        JudgeOutput(**kwargs)


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        JudgeOutput(pairwise_choice=PairwiseChoice.B, confidence=1.5)


def test_usage_non_negative() -> None:
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=-1)


def test_trial_round_trip_with_timestamp() -> None:
    trial = Trial(
        run_fingerprint="jl1:abc",
        example_id="ex-1",
        repeat_index=0,
        order=("a", "b"),
        request_hash="req-1",
        output=JudgeOutput(pairwise_choice=PairwiseChoice.B),
        created_at=FIXED_TIME,
    )
    restored = Trial.model_validate(trial.model_dump(mode="json"))
    assert restored == trial
    assert restored.created_at == FIXED_TIME


def test_trial_requires_non_empty_order() -> None:
    with pytest.raises(ValidationError):
        Trial(
            run_fingerprint="jl1:abc",
            example_id="ex",
            repeat_index=0,
            order=(),
            request_hash="r",
            output=JudgeOutput(),
        )


def test_trial_repeat_index_non_negative() -> None:
    with pytest.raises(ValidationError):
        Trial(
            run_fingerprint="jl1:abc",
            example_id="ex",
            repeat_index=-1,
            order=("a",),
            request_hash="r",
            output=JudgeOutput(),
        )


def test_evaluation_run_defaults_pending() -> None:
    run = EvaluationRun(config=_config(), n_examples=40)
    assert run.status is RunStatus.PENDING
    restored = EvaluationRun.model_validate(run.model_dump(mode="json"))
    assert restored == run


def test_evaluation_run_negative_examples_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluationRun(config=_config(), n_examples=-1)


def test_human_label_pairwise_only_valid() -> None:
    label = HumanLabel(example_id="ex", annotator_id="ann-1", pairwise_choice=PairwiseChoice.TIE)
    assert label.pairwise_choice is PairwiseChoice.TIE


def test_human_label_pointwise_only_valid() -> None:
    label = HumanLabel(example_id="ex", annotator_id="ann-1", pointwise_score=0.5)
    assert label.pointwise_score == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither verdict
        {"pairwise_choice": PairwiseChoice.A, "pointwise_score": 0.5},  # both
    ],
)
def test_human_label_requires_exactly_one_verdict(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        HumanLabel(example_id="ex", annotator_id="ann-1", **kwargs)
