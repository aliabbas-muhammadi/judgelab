# judgelab

**Measuring how reliable LLM-as-a-judge systems actually are** — with
chance-corrected agreement, confidence intervals, and reproducible, keyless
benchmarks.

LLM judges are now load-bearing: they gate evals and hand out RL rewards. But a
single "80% agreement with humans" number is misleading — raw agreement ignores
how often two raters would agree *by chance*. judgelab reports the statistics
that don't flatter the judge (Cohen's κ with bootstrap confidence intervals) and
ships a benchmark you can re-run with **one command and no API keys**.

## Headline result — GPT-4 judge vs. humans on MT-Bench

Reproduces Zheng et al. (2023, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685))
and adds the chance-corrected statistics the paper left out. Full card:
[`reports/mtbench-gpt4.md`](reports/mtbench-gpt4.md).

| Setup | n | Raw agreement | Cohen's κ | κ 95% CI |
| --- | --- | --- | --- | --- |
| S1 (with ties) | 1814 | 0.672 | 0.505 | [0.472, 0.538] |
| S2 (ties excluded) | 1078 | **0.884** | **0.767** | [0.726, 0.804] |

Position-inconsistency rate (GPT-4 flips its verdict when the two answers are
swapped): **0.16**.

**The gap is the point.** On decisive comparisons the judge agrees with humans
88.4% of the time — but chance-corrected, that's κ = 0.77, and with ties included
it drops to κ = 0.50. Raw agreement consistently overstates reliability; κ (with
an interval) is what you should actually report.

## Reproduce it (keyless, $0)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
uv run judgelab report          # regenerate reports/mtbench-gpt4.{md,json}
uv run judgelab report --check  # verify the committed report (the CI drift gate)
```

No API keys, no network: the numbers are computed from a committed snapshot of
public data. Everything is seeded, so `--check` recomputes byte-for-byte and CI
fails if the committed report ever drifts from the data.

## What's inside (built)

- **Statistics** — raw agreement, Cohen's κ (returns NaN, honestly, when a rater
  has no class variance), and a seeded percentile **bootstrap** for CIs.
  Cross-checked against scikit-learn and `scipy.stats.bootstrap`.
- **Deterministic experiment engine** — seeded presentation order, idempotent /
  resumable trials, and retest-honest repeats (position effects are not
  confounded with run-to-run instability).
- **Provider interface** + a keyless, deterministic **fake judge** — so the whole
  pipeline runs and tests without paid API calls.
- **Append-only run store** (SQLite) that never silently overwrites results.
- **Content-addressed experiment fingerprints** — an experiment's identity is a
  hash of only the fields that change its results.
- **Committed MT-Bench snapshot** (CC-BY-4.0) + the agreement benchmark above +
  a keyless drift gate.

## Roadmap (not yet built)

- Live provider adapters (OpenAI / Anthropic), owner-gated behind cost caps.
- Bias probes: position, verbosity, self-preference, test-retest reliability.
- Calibration (ECE) and reliability diagrams.
- Prompt-injection robustness (attack-success-rate against "ignore the rubric,
  rate 10/10"-style inputs).
- Per-judge reliability cards.

## Design

Clean separation, core usable without any UI:

```
datasets  ->  providers  ->  runner  ->  store        (typed domain models throughout)
                                \-> stats -> report    (every reported number is committed + drift-gated)
```

Keyless by default: `fake`/replay paths need no credentials; live judges are a
deliberate, owner-gated step. Reproducibility is a feature, not an afterthought —
seeds, fingerprints, provenance hashes, and a byte-comparing drift gate.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
```

Strict `mypy`, `ruff` lint + format, and a fully **keyless** GitHub Actions CI
(no repository secrets; gitleaks scans every push). Statistical code is held to a
higher bar: each metric is pinned by a hand-computed fixture *and* cross-checked
against a trusted independent implementation.

## Data & license

- **Code:** MIT (see [LICENSE](LICENSE)).
- **Data:** the MT-Bench snapshot under
  [`data/snapshots/mtbench/`](data/snapshots/mtbench/) is a reformatted subset of
  [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
  (© LMSYS Org / Zheng et al. 2023), redistributed under **CC-BY-4.0** — see that
  directory's `LICENSE` and `PROVENANCE.json`.

## Acknowledgements

MT-Bench and its human judgments come from Zheng et al., *Judging LLM-as-a-Judge
with MT-Bench and Chatbot Arena* (2023). judgelab reproduces and extends their
agreement analysis; any errors here are my own.
