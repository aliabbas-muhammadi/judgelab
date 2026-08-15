"""Validation and round-trip tests for the config-side domain models."""

import hashlib

import pytest
from pydantic import ValidationError

from judgelab.types import (
    CandidateResponse,
    Example,
    ExperimentConfig,
    Judge,
    JudgeVersion,
    Protocol,
    SamplingParams,
    TaskType,
)

DATASET_HASH = hashlib.sha256(b"demo-dataset").hexdigest()


def _candidate(cid: str, text: str = "an answer") -> CandidateResponse:
    return CandidateResponse(id=cid, text=text)


def _pairwise_example() -> Example:
    return Example(
        id="ex-1",
        task_type=TaskType.PAIRWISE,
        prompt="Which answer is better?",
        candidates=(_candidate("a"), _candidate("b")),
    )


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="mtbench pairwise smoke",
        dataset_id="mtbench",
        dataset_hash=DATASET_HASH,
        judge=Judge(
            id="gpt4-pairwise",
            provider="replay",
            protocol=Protocol.PAIRWISE,
            version=JudgeVersion(
                model="gpt-4",
                prompt_template_version="pairwise-v1",
                sampling=SamplingParams(temperature=0.0, max_tokens=120),
            ),
        ),
        seed=7,
        n_repeats=5,
    )


def test_pairwise_example_valid() -> None:
    ex = _pairwise_example()
    assert len(ex.candidates) == 2


def test_pointwise_example_valid() -> None:
    ex = Example(
        id="ex-2",
        task_type=TaskType.POINTWISE,
        prompt="Rate this answer.",
        candidates=(_candidate("only"),),
    )
    assert len(ex.candidates) == 1


@pytest.mark.parametrize(
    ("task_type", "n"),
    [(TaskType.PAIRWISE, 1), (TaskType.PAIRWISE, 3), (TaskType.POINTWISE, 2)],
)
def test_wrong_candidate_count_rejected(task_type: TaskType, n: int) -> None:
    with pytest.raises(ValidationError, match="candidate"):
        Example(
            id="bad",
            task_type=task_type,
            prompt="p",
            candidates=tuple(_candidate(str(i)) for i in range(n)),
        )


def test_duplicate_candidate_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate candidate ids"):
        Example(
            id="dup",
            task_type=TaskType.PAIRWISE,
            prompt="p",
            candidates=(_candidate("x"), _candidate("x")),
        )


def test_empty_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _candidate("")


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        CandidateResponse(id="a", text="t", surprise="nope")  # type: ignore[call-arg]


def test_models_are_frozen() -> None:
    cand = _candidate("a")
    with pytest.raises(ValidationError):
        cand.text = "mutated"  # type: ignore[misc]


def test_sampling_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        SamplingParams(temperature=3.0)
    with pytest.raises(ValidationError):
        SamplingParams(top_p=1.5)
    with pytest.raises(ValidationError):
        SamplingParams(max_tokens=0)


def test_unset_sampling_is_none() -> None:
    sp = SamplingParams()
    assert sp.temperature is None
    assert sp.max_tokens is None


@pytest.mark.parametrize("bad", ["", "xyz", "0x" + "a" * 62, "a" * 63, "g" * 64])
def test_dataset_hash_must_be_sha256_hex(bad: str) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(name="n", dataset_id="d", dataset_hash=bad, judge=_config().judge)


def test_dataset_hash_normalised_to_lowercase() -> None:
    cfg = ExperimentConfig(
        name="n",
        dataset_id="d",
        dataset_hash=DATASET_HASH.upper(),
        judge=_config().judge,
    )
    assert cfg.dataset_hash == DATASET_HASH


def test_n_repeats_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="n",
            dataset_id="d",
            dataset_hash=DATASET_HASH,
            judge=_config().judge,
            n_repeats=0,
        )


def test_json_round_trip_preserves_config() -> None:
    cfg = _config()
    dumped = cfg.model_dump(mode="json")
    restored = ExperimentConfig.model_validate(dumped)
    assert restored == cfg


def test_json_round_trip_preserves_example() -> None:
    ex = _pairwise_example()
    restored = Example.model_validate(ex.model_dump(mode="json"))
    assert restored == ex
