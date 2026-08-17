"""Typed, immutable domain models for judgelab experiments.

This module defines the *config / input* side of the data model — the entities
an experiment is defined over (examples, candidate responses, judges) and the
experiment configuration itself, plus the result-side entities an experiment
produces (judge outputs, trials, runs, human labels). Report-side aggregates
(agreement metrics, reliability cards) arrive when the stats layer lands.

Every model is frozen and rejects unknown fields, so a typo in a config file
fails loudly rather than being silently ignored. Enums are string-valued so they
serialise stably into the canonical JSON used for experiment fingerprinting.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_HEX = r"^[0-9a-fA-F]{64}$"


class TaskType(enum.StrEnum):
    """The kind of judgement an example calls for."""

    PAIRWISE = "pairwise"
    POINTWISE = "pointwise"


class Protocol(enum.StrEnum):
    """How a judge renders its verdict."""

    PAIRWISE = "pairwise"
    POINTWISE = "pointwise"


class _Frozen(BaseModel):
    """Base for immutable, strict domain models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CandidateResponse(_Frozen):
    """A single response to be judged."""

    id: str = Field(min_length=1)
    text: str
    # The model that produced the response, when known. Never shown to a judge
    # during blind evaluation; used only for self-preference analysis.
    model: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Example(_Frozen):
    """One evaluation item: a prompt, its candidate response(s), optional reference."""

    id: str = Field(min_length=1)
    task_type: TaskType
    prompt: str
    candidates: tuple[CandidateResponse, ...]
    reference: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_candidates(self) -> Self:
        expected = 2 if self.task_type is TaskType.PAIRWISE else 1
        if len(self.candidates) != expected:
            raise ValueError(
                f"{self.task_type.value} example {self.id!r} needs exactly "
                f"{expected} candidate(s), got {len(self.candidates)}"
            )
        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"example {self.id!r} has duplicate candidate ids: {ids}")
        return self


class SamplingParams(_Frozen):
    """Judge sampling settings.

    ``None`` means "not set" — the provider adapter must send nothing for that
    parameter, so a judge is probed exactly as deployed (some production judges
    pass no temperature at all, and adding one would change what is measured).
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None


class JudgeVersion(_Frozen):
    """The pinned identity of a judge — everything that, if changed, changes behaviour."""

    model: str = Field(min_length=1)
    prompt_template_version: str = Field(min_length=1)
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    # Provider-reported model snapshot, when exposed (e.g. "gpt-4o-2024-08-06").
    model_version: str | None = None


class Judge(_Frozen):
    """A judge under test."""

    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    protocol: Protocol
    version: JudgeVersion
    metadata: dict[str, str] = Field(default_factory=dict)


class ExperimentConfig(_Frozen):
    """A reproducible experiment definition.

    Two configs that share the same identity fields must produce the same
    fingerprint (computed in the fingerprint module); display / provenance fields
    such as ``name`` and ``notes`` are deliberately outside that identity.
    """

    name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_hash: str = Field(pattern=SHA256_HEX)
    judge: Judge
    seed: int = 0
    n_repeats: int = Field(default=1, ge=1)
    randomize_order: bool = True
    code_version: str | None = None
    notes: str | None = None

    @field_validator("dataset_hash")
    @classmethod
    def _normalise_hash(cls, value: str) -> str:
        return value.lower()


class Usage(_Frozen):
    """Token usage reported by a provider, when available."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class PairwiseChoice(enum.StrEnum):
    """Which presented slot a pairwise judgement preferred (A = first, B = second)."""

    A = "A"
    B = "B"
    TIE = "tie"


class RunStatus(enum.StrEnum):
    """Lifecycle state of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"


class JudgeOutput(_Frozen):
    """A judge's normalised result for a single judgement.

    Parse failures and provider errors are recorded as data, never silently
    dropped: a failed output carries ``parse_error`` / ``error`` and no verdict,
    so refusal and parse-failure rates remain measurable.
    """

    raw_text: str | None = None
    pairwise_choice: PairwiseChoice | None = None
    pointwise_score: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    parse_error: bool = False
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _failed_output_has_no_verdict(self) -> Self:
        failed = self.parse_error or self.error is not None
        has_verdict = self.pairwise_choice is not None or self.pointwise_score is not None
        if failed and has_verdict:
            raise ValueError("a failed judge output must not also carry a verdict")
        return self


class Trial(_Frozen):
    """One judgement: a single example judged under one presentation order and repeat.

    ``order`` is the candidate ids in the order they were presented to the judge,
    so a pairwise slot choice (A/B) can be mapped back to a candidate and position
    bias measured. ``request_hash`` keys idempotent, resumable persistence.
    """

    run_fingerprint: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    repeat_index: int = Field(ge=0)
    order: tuple[str, ...] = Field(min_length=1)
    request_hash: str = Field(min_length=1)
    output: JudgeOutput
    created_at: datetime | None = None


class EvaluationRun(_Frozen):
    """Metadata for one experiment run.

    Its identity is ``fingerprint(config)``, computed by the store rather than
    stored redundantly here so the two can never disagree.
    """

    config: ExperimentConfig
    n_examples: int = Field(ge=0)
    status: RunStatus = RunStatus.PENDING
    created_at: datetime | None = None


class HumanLabel(_Frozen):
    """A human annotator's judgement on one example (ground truth for agreement)."""

    example_id: str = Field(min_length=1)
    annotator_id: str = Field(min_length=1)
    pairwise_choice: PairwiseChoice | None = None
    pointwise_score: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _exactly_one_verdict(self) -> Self:
        has_pair = self.pairwise_choice is not None
        has_point = self.pointwise_score is not None
        if has_pair == has_point:
            raise ValueError("HumanLabel needs exactly one of pairwise_choice or pointwise_score")
        return self


__all__ = [
    "CandidateResponse",
    "EvaluationRun",
    "Example",
    "ExperimentConfig",
    "HumanLabel",
    "Judge",
    "JudgeOutput",
    "JudgeVersion",
    "PairwiseChoice",
    "Protocol",
    "RunStatus",
    "SamplingParams",
    "TaskType",
    "Trial",
    "Usage",
]
