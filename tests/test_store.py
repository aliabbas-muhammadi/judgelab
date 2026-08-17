"""Tests for the append-only, conflict-safe run store."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from judgelab.fingerprint import fingerprint
from judgelab.store import RunStore, StoreConflictError
from judgelab.types import (
    EvaluationRun,
    ExperimentConfig,
    Judge,
    JudgeOutput,
    JudgeVersion,
    PairwiseChoice,
    Protocol,
    RunStatus,
    Trial,
)

FIXED_TIME = datetime(2026, 8, 16, 12, 0, 0, tzinfo=UTC)


def _config(name: str = "run-a") -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        dataset_id="mtbench",
        dataset_hash=hashlib.sha256(b"x").hexdigest(),
        judge=Judge(
            id="j",
            provider="fake",
            protocol=Protocol.PAIRWISE,
            version=JudgeVersion(model="m", prompt_template_version="v1"),
        ),
    )


def _run(name: str = "run-a", n: int = 2, status: RunStatus = RunStatus.PENDING) -> EvaluationRun:
    return EvaluationRun(config=_config(name), n_examples=n, status=status, created_at=FIXED_TIME)


def _trial(
    run_fp: str, request_hash: str = "req-1", choice: PairwiseChoice = PairwiseChoice.A
) -> Trial:
    return Trial(
        run_fingerprint=run_fp,
        example_id="ex",
        repeat_index=0,
        order=("a", "b"),
        request_hash=request_hash,
        output=JudgeOutput(pairwise_choice=choice),
    )


def test_save_and_load_run_round_trip() -> None:
    store = RunStore()
    run = _run()
    fp = store.save_run(run)
    assert fp == fingerprint(run.config)
    assert store.has_run(fp)
    assert store.load_run(fp) == run


def test_load_missing_run_is_none() -> None:
    assert RunStore().load_run("jl1:missing") is None


def test_save_run_is_idempotent() -> None:
    store = RunStore()
    run = _run()
    assert store.save_run(run) == store.save_run(run)


def test_save_run_conflict_on_same_fingerprint_different_config() -> None:
    store = RunStore()
    a, b = _run(name="run-a"), _run(name="run-b")
    # name is excluded from the fingerprint, so these share an identity...
    assert fingerprint(a.config) == fingerprint(b.config)
    store.save_run(a)
    # ...but the stored metadata differs, so overwriting is refused.
    with pytest.raises(StoreConflictError):
        store.save_run(b)
    assert store.load_run(fingerprint(a.config)) == a


def test_set_status_updates() -> None:
    store = RunStore()
    fp = store.save_run(_run(status=RunStatus.PENDING))
    store.set_status(fp, RunStatus.COMPLETE)
    loaded = store.load_run(fp)
    assert loaded is not None
    assert loaded.status is RunStatus.COMPLETE


def test_set_status_missing_run_raises() -> None:
    with pytest.raises(KeyError):
        RunStore().set_status("jl1:missing", RunStatus.COMPLETE)


def test_save_and_load_trials() -> None:
    store = RunStore()
    fp = store.save_run(_run())
    store.save_trial(_trial(fp, "req-1"))
    store.save_trial(_trial(fp, "req-2"))
    assert store.count_trials(fp) == 2
    assert store.has_trial(fp, "req-1")
    assert {t.request_hash for t in store.load_trials(fp)} == {"req-1", "req-2"}


def test_save_trial_is_idempotent() -> None:
    store = RunStore()
    fp = store.save_run(_run())
    trial = _trial(fp, "req-1")
    store.save_trial(trial)
    store.save_trial(trial)
    assert store.count_trials(fp) == 1


def test_save_trial_conflict_preserves_original() -> None:
    store = RunStore()
    fp = store.save_run(_run())
    store.save_trial(_trial(fp, "req-1", choice=PairwiseChoice.A))
    with pytest.raises(StoreConflictError):
        store.save_trial(_trial(fp, "req-1", choice=PairwiseChoice.B))
    assert store.count_trials(fp) == 1
    assert store.load_trials(fp)[0].output.pairwise_choice is PairwiseChoice.A


def test_persists_across_connections(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "runs.db"  # nested path also exercises parent-dir creation
    with RunStore(db) as store:
        fp = store.save_run(_run())
        store.save_trial(_trial(fp))
    with RunStore(db) as store:
        assert store.has_run(fp)
        assert store.count_trials(fp) == 1
