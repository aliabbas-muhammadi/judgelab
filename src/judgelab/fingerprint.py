"""Content-addressed experiment fingerprints for reproducibility.

An experiment's *fingerprint* is a stable hash over only the fields that
determine its results — the dataset snapshot, the judge's pinned identity, and
the run parameters (seed, repeats, ordering). Display / provenance fields
(``name``, ``notes``, timestamps, run ids) are deliberately excluded, so
annotating an experiment never changes its identity, while a later field added
to :class:`~judgelab.types.ExperimentConfig` cannot silently churn the hash.

Recipe: pydantic ``model_dump(mode="json")`` -> canonical JSON (sorted keys, no
whitespace, ASCII, NaN/Inf refused) -> sha256. The result is prefixed with a
scheme tag (``jl1:``) so the scheme can evolve without colliding with older
fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from judgelab.types import ExperimentConfig, Protocol, SamplingParams

SCHEME = "jl1"


class FingerprintInput(BaseModel):
    """The explicit, closed set of fields that determine an experiment's results.

    Kept separate from :class:`~judgelab.types.ExperimentConfig` so that display
    and provenance fields on the config are structurally excluded from identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_hash: str
    judge_id: str
    provider: str
    protocol: Protocol
    model: str
    model_version: str | None
    prompt_template_version: str
    sampling: SamplingParams
    seed: int
    n_repeats: int = Field(ge=1)
    randomize_order: bool
    code_version: str | None

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> Self:
        judge = config.judge
        version = judge.version
        return cls(
            dataset_hash=config.dataset_hash,
            judge_id=judge.id,
            provider=judge.provider,
            protocol=judge.protocol,
            model=version.model,
            model_version=version.model_version,
            prompt_template_version=version.prompt_template_version,
            sampling=version.sampling,
            seed=config.seed,
            n_repeats=config.n_repeats,
            randomize_order=config.randomize_order,
            code_version=config.code_version,
        )


def canonical_json(fp_input: FingerprintInput) -> bytes:
    """Serialise fingerprint inputs to canonical, deterministic JSON bytes.

    ``sort_keys`` removes dict-ordering nondeterminism (recursively), the compact
    separators remove whitespace drift, and ``allow_nan=False`` refuses NaN/Inf,
    which are neither valid JSON nor stable hash inputs.
    """
    payload = fp_input.model_dump(mode="json")
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def fingerprint(config: ExperimentConfig) -> str:
    """Return the versioned, content-addressed fingerprint of an experiment config."""
    digest = hashlib.sha256(canonical_json(FingerprintInput.from_config(config))).hexdigest()
    return f"{SCHEME}:{digest}"


__all__ = ["SCHEME", "FingerprintInput", "canonical_json", "fingerprint"]
