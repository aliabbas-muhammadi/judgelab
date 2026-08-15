"""Deterministic golden-hash tests for experiment fingerprinting.

The canonical JSON is pinned against a hand-authored spec string, so any change
to the field set, key ordering, null handling, float repr, or enum values fails
loudly rather than silently producing a different hash.
"""

import hashlib
from typing import Any

import pytest

from judgelab.fingerprint import SCHEME, FingerprintInput, canonical_json, fingerprint
from judgelab.types import (
    ExperimentConfig,
    Judge,
    JudgeVersion,
    Protocol,
    SamplingParams,
)

DATASET_HASH = hashlib.sha256(b"demo-dataset").hexdigest()

# The exact canonical serialisation the golden config must produce. Authored by
# hand as the spec; keys are sorted recursively, no whitespace, nulls explicit.
EXPECTED_CANONICAL = (
    "{"
    '"code_version":null,'
    f'"dataset_hash":"{DATASET_HASH}",'
    '"judge_id":"gpt4-pairwise",'
    '"model":"gpt-4",'
    '"model_version":null,'
    '"n_repeats":5,'
    '"prompt_template_version":"pairwise-v1",'
    '"protocol":"pairwise",'
    '"provider":"replay",'
    '"randomize_order":true,'
    '"sampling":{"max_tokens":120,"seed":null,"temperature":0.0,"top_p":null},'
    '"seed":7'
    "}"
)


def _config(**overrides: Any) -> ExperimentConfig:
    """Build the golden ExperimentConfig, with optional field overrides."""
    o = {
        "name": "golden experiment",
        "notes": None,
        "dataset_id": "mtbench",
        "dataset_hash": DATASET_HASH,
        "judge_id": "gpt4-pairwise",
        "provider": "replay",
        "protocol": Protocol.PAIRWISE,
        "model": "gpt-4",
        "model_version": None,
        "prompt_template_version": "pairwise-v1",
        "temperature": 0.0,
        "max_tokens": 120,
        "seed": 7,
        "n_repeats": 5,
        "randomize_order": True,
        "code_version": None,
        **overrides,
    }
    return ExperimentConfig(
        name=o["name"],
        notes=o["notes"],
        dataset_id=o["dataset_id"],
        dataset_hash=o["dataset_hash"],
        seed=o["seed"],
        n_repeats=o["n_repeats"],
        randomize_order=o["randomize_order"],
        code_version=o["code_version"],
        judge=Judge(
            id=o["judge_id"],
            provider=o["provider"],
            protocol=o["protocol"],
            version=JudgeVersion(
                model=o["model"],
                model_version=o["model_version"],
                prompt_template_version=o["prompt_template_version"],
                sampling=SamplingParams(temperature=o["temperature"], max_tokens=o["max_tokens"]),
            ),
        ),
    )


def test_canonical_json_matches_hand_written_spec() -> None:
    fp_input = FingerprintInput.from_config(_config())
    assert canonical_json(fp_input).decode() == EXPECTED_CANONICAL


def test_golden_fingerprint_is_stable() -> None:
    expected = f"{SCHEME}:" + hashlib.sha256(EXPECTED_CANONICAL.encode()).hexdigest()
    assert fingerprint(_config()) == expected


def test_fingerprint_shape() -> None:
    fp = fingerprint(_config())
    scheme, _, digest = fp.partition(":")
    assert scheme == SCHEME
    assert len(digest) == 64
    assert int(digest, 16) >= 0  # valid hex


def test_deterministic_across_rebuilds() -> None:
    assert fingerprint(_config()) == fingerprint(_config())


# Fields that determine results — changing any must change the fingerprint.
IDENTITY_OVERRIDES = [
    {"dataset_hash": hashlib.sha256(b"other").hexdigest()},
    {"judge_id": "gpt4-pairwise-v2"},
    {"provider": "openai"},
    {"protocol": Protocol.POINTWISE},
    {"model": "gpt-4o"},
    {"model_version": "gpt-4-0613"},
    {"prompt_template_version": "pairwise-v2"},
    {"temperature": 0.7},
    {"max_tokens": 256},
    {"seed": 8},
    {"n_repeats": 6},
    {"randomize_order": False},
    {"code_version": "abc123"},
]


@pytest.mark.parametrize("override", IDENTITY_OVERRIDES, ids=lambda o: next(iter(o)))
def test_identity_fields_change_fingerprint(override: dict[str, Any]) -> None:
    assert fingerprint(_config(**override)) != fingerprint(_config())


# Display / provenance fields — changing any must NOT change the fingerprint.
EXCLUDED_OVERRIDES = [
    {"name": "a totally different name"},
    {"notes": "some provenance note"},
    {"dataset_id": "different-label-same-content"},
]


@pytest.mark.parametrize("override", EXCLUDED_OVERRIDES, ids=lambda o: next(iter(o)))
def test_display_fields_do_not_change_fingerprint(override: dict[str, Any]) -> None:
    assert fingerprint(_config(**override)) == fingerprint(_config())


def test_dataset_hash_case_does_not_change_fingerprint() -> None:
    # ExperimentConfig lowercases the hash, so upper/lower are the same experiment.
    assert fingerprint(_config(dataset_hash=DATASET_HASH.upper())) == fingerprint(_config())
