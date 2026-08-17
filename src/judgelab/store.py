"""Append-only persistence for experiment runs and their trials.

The store is deliberately conservative: it **never silently overwrites** prior
results. Re-saving identical content is an idempotent no-op (so runs resume by
``request_hash``), but saving a different payload under a key that already exists
raises :class:`StoreConflictError` rather than clobbering the earlier record.

Runs are keyed by their experiment fingerprint; trials are keyed by
``(run_fingerprint, request_hash)``. Only ``status`` is mutable (the run
lifecycle) — config and trial content are immutable once written.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from judgelab.fingerprint import fingerprint
from judgelab.types import EvaluationRun, ExperimentConfig, RunStatus, Trial

MEMORY = ":memory:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    fingerprint TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    n_examples  INTEGER NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT
);
CREATE TABLE IF NOT EXISTS trials (
    run_fingerprint TEXT NOT NULL,
    request_hash    TEXT NOT NULL,
    trial_json      TEXT NOT NULL,
    PRIMARY KEY (run_fingerprint, request_hash)
);
"""


class StoreConflictError(RuntimeError):
    """Raised when a key already holds different content — never overwrite it."""


class RunStore:
    """SQLite-backed, append-only store for runs and trials."""

    def __init__(self, path: str | Path = MEMORY) -> None:
        path_str = str(path)
        if path_str != MEMORY:
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path_str)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- runs -------------------------------------------------------------

    def save_run(self, run: EvaluationRun) -> str:
        """Persist a run and return its fingerprint. Idempotent; conflict-safe."""
        fp = fingerprint(run.config)
        config_json = run.config.model_dump_json()
        existing = self._conn.execute(
            "SELECT config_json FROM runs WHERE fingerprint = ?", (fp,)
        ).fetchone()
        if existing is not None:
            if existing["config_json"] != config_json:
                raise StoreConflictError(
                    f"run {fp} already stored with a different config; refusing to overwrite"
                )
            return fp
        created = run.created_at.isoformat() if run.created_at is not None else None
        self._conn.execute(
            "INSERT INTO runs (fingerprint, config_json, n_examples, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fp, config_json, run.n_examples, run.status.value, created),
        )
        self._conn.commit()
        return fp

    def set_status(self, fingerprint_: str, status: RunStatus) -> None:
        cursor = self._conn.execute(
            "UPDATE runs SET status = ? WHERE fingerprint = ?", (status.value, fingerprint_)
        )
        if cursor.rowcount == 0:
            raise KeyError(f"no run with fingerprint {fingerprint_!r}")
        self._conn.commit()

    def has_run(self, fingerprint_: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM runs WHERE fingerprint = ?", (fingerprint_,)
        ).fetchone()
        return row is not None

    def load_run(self, fingerprint_: str) -> EvaluationRun | None:
        row = self._conn.execute(
            "SELECT config_json, n_examples, status, created_at FROM runs WHERE fingerprint = ?",
            (fingerprint_,),
        ).fetchone()
        if row is None:
            return None
        created = row["created_at"]
        return EvaluationRun(
            config=ExperimentConfig.model_validate_json(row["config_json"]),
            n_examples=row["n_examples"],
            status=RunStatus(row["status"]),
            created_at=datetime.fromisoformat(created) if created is not None else None,
        )

    # --- trials -----------------------------------------------------------

    def save_trial(self, trial: Trial) -> None:
        """Persist one trial. Idempotent by (run, request_hash); conflict-safe."""
        trial_json = trial.model_dump_json()
        existing = self._conn.execute(
            "SELECT trial_json FROM trials WHERE run_fingerprint = ? AND request_hash = ?",
            (trial.run_fingerprint, trial.request_hash),
        ).fetchone()
        if existing is not None:
            if existing["trial_json"] != trial_json:
                raise StoreConflictError(
                    f"trial {trial.request_hash!r} already stored with different content; "
                    "refusing to overwrite"
                )
            return
        self._conn.execute(
            "INSERT INTO trials (run_fingerprint, request_hash, trial_json) VALUES (?, ?, ?)",
            (trial.run_fingerprint, trial.request_hash, trial_json),
        )
        self._conn.commit()

    def has_trial(self, run_fingerprint: str, request_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM trials WHERE run_fingerprint = ? AND request_hash = ?",
            (run_fingerprint, request_hash),
        ).fetchone()
        return row is not None

    def count_trials(self, run_fingerprint: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM trials WHERE run_fingerprint = ?", (run_fingerprint,)
        ).fetchone()
        return int(row["n"])

    def load_trials(self, run_fingerprint: str) -> list[Trial]:
        rows = self._conn.execute(
            "SELECT trial_json FROM trials WHERE run_fingerprint = ? ORDER BY request_hash",
            (run_fingerprint,),
        ).fetchall()
        return [Trial.model_validate_json(row["trial_json"]) for row in rows]


__all__ = ["MEMORY", "RunStore", "StoreConflictError"]
