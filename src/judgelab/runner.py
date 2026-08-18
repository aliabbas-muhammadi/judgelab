"""The deterministic experiment engine.

Given a config, a set of examples, a judge provider, and a store, the runner
judges every example ``n_repeats`` times under a seeded presentation order and
persists each judgement as a :class:`~judgelab.types.Trial`. It is:

- **deterministic** — presentation order is derived from ``(config.seed,
  example id)`` via sha256, so a run reproduces on any machine (given a
  deterministic provider);
- **idempotent / resumable** — each trial has a ``request_hash`` that includes
  the repeat index; already-stored trials are skipped, so an interrupted run
  resumes without re-judging or double-counting;
- **retest-honest** — repeats reuse the *same* order (only the repeat index
  changes), so test-retest instability is not confounded with position effects.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from datetime import datetime

from judgelab.providers.base import JudgeProvider, JudgeRequest
from judgelab.store import RunStore
from judgelab.types import EvaluationRun, Example, ExperimentConfig, RunStatus, Trial

_SEED_BYTES = 8


def _presentation_order(config: ExperimentConfig, example: Example) -> tuple[str, ...]:
    """Deterministic candidate presentation order for an example."""
    ids = tuple(candidate.id for candidate in example.candidates)
    if not config.randomize_order or len(ids) < 2:
        return ids
    digest = hashlib.sha256(f"{config.seed}:{example.id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:_SEED_BYTES], "big"))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return tuple(shuffled)


def _request_hash(
    run_fingerprint: str, example_id: str, order: tuple[str, ...], repeat_index: int
) -> str:
    """Stable idempotence key. Includes the repeat index so retest calls are distinct."""
    payload = json.dumps(
        {
            "run": run_fingerprint,
            "example": example_id,
            "order": list(order),
            "repeat": repeat_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _check_protocol(config: ExperimentConfig, examples: Sequence[Example]) -> None:
    protocol = config.judge.protocol.value
    for example in examples:
        if example.task_type.value != protocol:
            raise ValueError(
                f"example {example.id!r} is {example.task_type.value} but the judge is {protocol}"
            )


def run_experiment(
    config: ExperimentConfig,
    examples: Sequence[Example],
    provider: JudgeProvider,
    store: RunStore,
    *,
    now: datetime | None = None,
) -> str:
    """Run an experiment to completion and return its fingerprint.

    Idempotent: re-invoking with the same inputs skips already-stored trials and
    adds nothing. ``now`` is an injected timestamp (never wall-clock) so runs
    stay reproducible.
    """
    _check_protocol(config, examples)
    run_fingerprint = store.save_run(
        EvaluationRun(
            config=config,
            n_examples=len(examples),
            status=RunStatus.RUNNING,
            created_at=now,
        )
    )
    for example in examples:
        order = _presentation_order(config, example)
        for repeat_index in range(config.n_repeats):
            request_hash = _request_hash(run_fingerprint, example.id, order, repeat_index)
            if store.has_trial(run_fingerprint, request_hash):
                continue
            output = provider.judge(JudgeRequest(example=example, judge=config.judge, order=order))
            store.save_trial(
                Trial(
                    run_fingerprint=run_fingerprint,
                    example_id=example.id,
                    repeat_index=repeat_index,
                    order=order,
                    request_hash=request_hash,
                    output=output,
                    created_at=now,
                )
            )
    store.set_status(run_fingerprint, RunStatus.COMPLETE)
    return run_fingerprint


__all__ = ["run_experiment"]
