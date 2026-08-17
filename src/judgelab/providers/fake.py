"""Deterministic, keyless fake judge for tests and pipeline development.

The fake produces a stable verdict from a sha256 of the judge identity, example
id, and presentation order, so a whole run is reproducible on any machine with
no API calls. The verdicts are arbitrary (hash-derived): the fake exercises the
pipeline and statistics plumbing — it does NOT measure any real judge.
"""

from __future__ import annotations

import hashlib

from judgelab.providers.base import JudgeProvider, JudgeRequest
from judgelab.types import JudgeOutput, PairwiseChoice, Protocol

_FIELD_SEP = "\x1f"


class FakeProvider(JudgeProvider):
    """A judge whose verdicts are a deterministic hash of the request.

    A ``salt`` shifts the verdicts deterministically, which is useful for
    simulating several distinct fake judges in a test.
    """

    def __init__(self, salt: str = "") -> None:
        self._salt = salt

    def _digest(self, request: JudgeRequest) -> bytes:
        version = request.judge.version
        parts = [
            self._salt,
            request.judge.id,
            request.judge.provider,
            version.model,
            version.model_version or "",
            version.prompt_template_version,
            request.example.id,
            *request.order,
        ]
        return hashlib.sha256(_FIELD_SEP.join(parts).encode("utf-8")).digest()

    def judge(self, request: JudgeRequest) -> JudgeOutput:
        digest = self._digest(request)
        confidence = digest[2] / 255.0
        if request.judge.protocol is Protocol.PAIRWISE:
            if digest[0] % 10 == 0:
                choice = PairwiseChoice.TIE
            elif digest[1] % 2 == 0:
                choice = PairwiseChoice.A
            else:
                choice = PairwiseChoice.B
            return JudgeOutput(
                pairwise_choice=choice,
                confidence=confidence,
                raw_text=f"fake:{choice.value}",
            )
        score = (int.from_bytes(digest[3:5], "big") % 1001) / 1000.0
        return JudgeOutput(
            pointwise_score=score,
            confidence=confidence,
            raw_text=f"fake:{score:.3f}",
        )


__all__ = ["FakeProvider"]
