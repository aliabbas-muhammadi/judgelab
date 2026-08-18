"""Tests for the MT-Bench snapshot loader and the committed data's integrity."""

import hashlib
import json
from pathlib import Path

from judgelab.datasets.mtbench import (
    MtbenchWinner,
    default_snapshot_dir,
    load_snapshot,
    load_votes,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "snapshots" / "mtbench"


def _provenance() -> dict[str, object]:
    return json.loads((SNAPSHOT / "PROVENANCE.json").read_text(encoding="utf-8"))


def test_committed_data_matches_provenance_hashes() -> None:
    files = _provenance()["files"]
    assert isinstance(files, dict)
    for name, meta in files.items():
        digest = hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest()
        assert digest == meta["sha256"], f"{name} content drifted from PROVENANCE"


def test_counts_match_provenance() -> None:
    files = _provenance()["files"]
    assert isinstance(files, dict)
    assert len(load_votes(SNAPSHOT / "human.jsonl")) == files["human.jsonl"]["rows"]
    assert len(load_votes(SNAPSHOT / "gpt4_pair.jsonl")) == files["gpt4_pair.jsonl"]["rows"]


def test_schema_is_well_formed() -> None:
    snapshot = load_snapshot(SNAPSHOT)
    votes = [*snapshot.human, *snapshot.gpt4_pair]
    assert votes  # non-empty
    assert all(isinstance(v.winner, MtbenchWinner) for v in votes)
    assert all(81 <= v.question_id <= 160 for v in votes)  # MT-Bench question id range
    assert all(v.turn in (1, 2) for v in votes)
    assert all(v.model_a != v.model_b for v in votes)


def test_default_dir_resolves_to_committed_snapshot() -> None:
    assert default_snapshot_dir() == SNAPSHOT


def test_split_judge_populations_differ() -> None:
    snapshot = load_snapshot(SNAPSHOT)
    human_judges = {v.judge for v in snapshot.human}
    gpt4_judges = {v.judge for v in snapshot.gpt4_pair}
    assert len(human_judges) > 1  # many human annotators
    assert "gpt4_pair" in gpt4_judges  # the GPT-4 judge tag
