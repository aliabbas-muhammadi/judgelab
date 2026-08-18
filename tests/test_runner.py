"""Tests for the deterministic experiment runner."""

import hashlib

import pytest

from judgelab.providers import FakeProvider
from judgelab.runner import run_experiment
from judgelab.store import RunStore
from judgelab.types import (
    CandidateResponse,
    Example,
    ExperimentConfig,
    Judge,
    JudgeVersion,
    Protocol,
    RunStatus,
    TaskType,
)


def _examples(n: int = 4, task: TaskType = TaskType.PAIRWISE) -> list[Example]:
    examples: list[Example] = []
    for i in range(n):
        if task is TaskType.PAIRWISE:
            candidates = (
                CandidateResponse(id="a", text=f"A{i}"),
                CandidateResponse(id="b", text=f"B{i}"),
            )
        else:
            candidates = (CandidateResponse(id="only", text=f"ans{i}"),)
        examples.append(Example(id=f"ex-{i}", task_type=task, prompt="p", candidates=candidates))
    return examples


def _config(
    *,
    seed: int = 0,
    n_repeats: int = 1,
    randomize: bool = True,
    protocol: Protocol = Protocol.PAIRWISE,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="run",
        dataset_id="d",
        dataset_hash=hashlib.sha256(b"x").hexdigest(),
        seed=seed,
        n_repeats=n_repeats,
        randomize_order=randomize,
        judge=Judge(
            id="j",
            provider="fake",
            protocol=protocol,
            version=JudgeVersion(model="m", prompt_template_version="v1"),
        ),
    )


def test_runs_to_completion_and_persists_every_trial() -> None:
    store = RunStore()
    fp = run_experiment(_config(n_repeats=2), _examples(4), FakeProvider(), store)
    run = store.load_run(fp)
    assert run is not None
    assert run.status is RunStatus.COMPLETE
    assert store.count_trials(fp) == 4 * 2


def test_idempotent_resume_adds_nothing() -> None:
    store = RunStore()
    cfg, examples, provider = _config(), _examples(4), FakeProvider()
    fp = run_experiment(cfg, examples, provider, store)
    before = store.load_trials(fp)
    fp_again = run_experiment(cfg, examples, provider, store)
    assert fp_again == fp
    assert store.load_trials(fp) == before
    assert store.count_trials(fp) == 4


def test_deterministic_across_stores() -> None:
    cfg, examples = _config(), _examples(4)
    a, b = RunStore(), RunStore()
    fp_a = run_experiment(cfg, examples, FakeProvider(), a)
    fp_b = run_experiment(cfg, examples, FakeProvider(), b)
    assert fp_a == fp_b
    assert a.load_trials(fp_a) == b.load_trials(fp_b)


def test_retest_repeats_share_one_order() -> None:
    store = RunStore()
    fp = run_experiment(_config(n_repeats=3), _examples(4), FakeProvider(), store)
    ex0 = [t for t in store.load_trials(fp) if t.example_id == "ex-0"]
    assert {t.repeat_index for t in ex0} == {0, 1, 2}
    assert len({t.order for t in ex0}) == 1  # same presentation order across repeats


def test_order_is_seeded_and_seed_sensitive() -> None:
    examples = _examples(8)

    def orders(seed: int) -> dict[str, tuple[str, ...]]:
        store = RunStore()
        fp = run_experiment(_config(seed=seed), examples, FakeProvider(), store)
        return {t.example_id: t.order for t in store.load_trials(fp)}

    assert orders(0) == orders(0)  # reproducible for a fixed seed
    assert orders(0) != orders(99)  # a different seed reshuffles at least one example


def test_no_randomize_keeps_declared_order() -> None:
    store = RunStore()
    fp = run_experiment(_config(randomize=False), _examples(4), FakeProvider(), store)
    assert all(t.order == ("a", "b") for t in store.load_trials(fp))


def test_protocol_mismatch_rejected() -> None:
    store = RunStore()
    with pytest.raises(ValueError, match="judge is pairwise"):
        run_experiment(
            _config(protocol=Protocol.PAIRWISE),
            _examples(2, task=TaskType.POINTWISE),
            FakeProvider(),
            store,
        )


def test_pointwise_run_scores_every_trial() -> None:
    store = RunStore()
    fp = run_experiment(
        _config(protocol=Protocol.POINTWISE),
        _examples(4, task=TaskType.POINTWISE),
        FakeProvider(),
        store,
    )
    trials = store.load_trials(fp)
    assert len(trials) == 4
    assert all(t.output.pointwise_score is not None for t in trials)
