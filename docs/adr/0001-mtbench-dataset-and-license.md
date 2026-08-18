# ADR 0001 — MT-Bench as the headline dataset, and its licensing

Status: accepted (2026-08-18)

## Context

judgelab's headline benchmark reproduces the MT-Bench GPT-4-judge vs human
agreement (Zheng et al. 2023, arXiv:2306.05685) and extends it with
chance-corrected agreement and confidence intervals. To keep CI keyless and the
result reproducible with zero API calls, the required data must be committed to
the repository — which is currently private but intended to become public, so
the data must be redistributable.

A primary-source license review (2026-08-18) established:

- The HuggingFace dataset `lmsys/mt_bench_human_judgments` is **CC-BY-4.0** and
  contains **both** the human pairwise votes (`human` split, 3355 rows) and the
  published GPT-4 pairwise verdicts (`gpt4_pair` split, 2400 rows), with an
  identical, self-contained 8-column schema.
- CC-BY-4.0 permits redistributing a subset (including commercially) with
  attribution; it is not ShareAlike or NonCommercial.
- FastChat (`lm-sys/FastChat`, Apache-2.0) does **not** commit the precomputed
  GPT-4 judgments on `main` (they are generated outputs; the paths 404), so the
  HF `gpt4_pair` split is the authoritative committed copy.

## Decision

- Use `lmsys/mt_bench_human_judgments` (both splits) as the headline dataset.
- Commit a **label/key-only** snapshot (`question_id`, `model_a`, `model_b`,
  `winner`, `judge`, `turn`); drop the embedded conversation texts. This keeps
  the snapshot small and sidesteps the residual question of redistributing raw
  model outputs, while fully preserving what the agreement metric needs.
- **Dual-license:** repository code stays under the repo's own LICENSE; the data
  directory carries its own CC-BY-4.0 `LICENSE` + `PROVENANCE.json` (source,
  revision, per-file sha256, citation, retrieval date, changes).
- Alignment for agreement joins `human` and `gpt4_pair` on
  `(question_id, model_a, model_b, turn)`; that logic lives in the reporting
  layer, not the loader.

## Consequences

- The MT-Bench reproduction is fully keyless and reproducible from committed data.
- Attribution obligations are met by the data-directory LICENSE + PROVENANCE.
- Running *new* (live) judges on MT-Bench later will require re-adding the
  conversation texts — a separate, deliberate extension.
- Position orderings (the license review's open question) are **resolved by the
  data itself**: the `gpt4_pair` split collapses the two orderings into one
  verdict per comparison and encodes a position flip as the winner value
  `"tie (inconsistent)"` (380 of 2400 rows). So position-consistency is already
  measurable from this column, without a separate `(b,a)` join.
