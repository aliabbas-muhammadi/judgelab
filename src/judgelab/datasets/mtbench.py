"""Loader for the committed MT-Bench agreement snapshot (CC-BY-4.0).

Reads the label/key snapshot built by ``scripts/build_mtbench_snapshot.py`` into
typed votes. Aligning human and GPT-4 verdicts (join, tie handling, position
analysis) and computing agreement live in the reporting layer, not here — this
module only loads and validates.

Data license: the snapshot under ``data/snapshots/mtbench/`` is CC-BY-4.0; see
that directory's LICENSE and PROVENANCE.json.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class MtbenchWinner(enum.StrEnum):
    """Which model in the ordered pair a verdict preferred.

    ``TIE_INCONSISTENT`` appears only in the GPT-4 split: it marks a comparison
    where the judge's verdict flipped between the two presentation orderings —
    the position-bias signal, encoded directly in the data.
    """

    MODEL_A = "model_a"
    MODEL_B = "model_b"
    TIE = "tie"
    TIE_INCONSISTENT = "tie (inconsistent)"


class MtbenchVote(BaseModel):
    """One pairwise verdict — a human annotator or the GPT-4 judge — on a comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: int
    model_a: str
    model_b: str
    winner: MtbenchWinner
    judge: str
    turn: int


@dataclass(frozen=True)
class MtbenchSnapshot:
    """The two aligned-able splits: human votes and published GPT-4 verdicts."""

    human: tuple[MtbenchVote, ...]
    gpt4_pair: tuple[MtbenchVote, ...]


def default_snapshot_dir() -> Path:
    """Repo-committed snapshot directory (resolved relative to this module)."""
    return Path(__file__).resolve().parents[3] / "data" / "snapshots" / "mtbench"


def load_votes(path: Path) -> list[MtbenchVote]:
    """Parse a snapshot JSONL file into validated votes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [MtbenchVote.model_validate_json(line) for line in lines if line.strip()]


def load_snapshot(directory: Path | None = None) -> MtbenchSnapshot:
    """Load both splits of the MT-Bench snapshot (defaults to the committed one)."""
    directory = directory or default_snapshot_dir()
    return MtbenchSnapshot(
        human=tuple(load_votes(directory / "human.jsonl")),
        gpt4_pair=tuple(load_votes(directory / "gpt4_pair.jsonl")),
    )


__all__ = [
    "MtbenchSnapshot",
    "MtbenchVote",
    "MtbenchWinner",
    "default_snapshot_dir",
    "load_snapshot",
    "load_votes",
]
