"""Provider interface: judges behind one normalising boundary.

A :class:`JudgeProvider` turns a :class:`JudgeRequest` into a normalised
:class:`~judgelab.types.JudgeOutput`. Keyless providers (fake, replay) and
owner-gated live providers (OpenAI, Anthropic) all implement the same interface,
so the runner and statistics never bind to a specific SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from judgelab.types import Example, Judge, JudgeOutput


class JudgeRequest(BaseModel):
    """Everything a provider needs to render one judgement.

    ``order`` is the candidate ids in the order they are presented to the judge;
    it must be a permutation of the example's candidates, so a pairwise slot
    choice can be mapped back to a candidate and position effects measured.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    example: Example
    judge: Judge
    order: tuple[str, ...]

    @model_validator(mode="after")
    def _order_is_permutation(self) -> Self:
        candidate_ids = {candidate.id for candidate in self.example.candidates}
        if len(self.order) != len(self.example.candidates) or set(self.order) != candidate_ids:
            raise ValueError("order must be a permutation of the example's candidate ids")
        return self


class JudgeProvider(ABC):
    """A judge behind the normalising boundary."""

    @abstractmethod
    def judge(self, request: JudgeRequest) -> JudgeOutput:
        """Return a normalised verdict for a single judgement request."""
        raise NotImplementedError


__all__ = ["JudgeProvider", "JudgeRequest"]
