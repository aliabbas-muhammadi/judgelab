"""Judge providers behind a single normalising interface."""

from judgelab.providers.base import JudgeProvider, JudgeRequest
from judgelab.providers.fake import FakeProvider

__all__ = ["FakeProvider", "JudgeProvider", "JudgeRequest"]
