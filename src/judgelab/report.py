"""Generate and verify the committed MT-Bench agreement report.

The rendered Markdown + JSON under ``reports/`` are committed artifacts. CI
recomputes them from the committed snapshot and fails if they drift (the keyless
drift gate), so the published numbers can never silently diverge from the data.
Everything is seeded, so the recompute is byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from judgelab.benchmarks import AgreementResult, Metric, compute_agreement
from judgelab.datasets.mtbench import load_snapshot

REPORT_SEED = 0
REPORT_RESAMPLES = 2000
MD_NAME = "mtbench-gpt4.md"
JSON_NAME = "mtbench-gpt4.json"


def reports_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "reports"


def build_result() -> AgreementResult:
    """Compute the agreement result under the canonical (seeded) report config."""
    return compute_agreement(load_snapshot(), n_resamples=REPORT_RESAMPLES, seed=REPORT_SEED)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _row(label: str, metric: Metric) -> str:
    ci = f"[{_fmt(metric.kappa_ci_low)}, {_fmt(metric.kappa_ci_high)}]"
    return (
        f"| {label} | {metric.n} | {_fmt(metric.raw_agreement)} | "
        f"{_fmt(metric.cohen_kappa)} | {ci} |"
    )


def render_markdown(result: AgreementResult) -> str:
    lines = [
        "# MT-Bench — GPT-4 judge vs human agreement",
        "",
        "Reproduction of Zheng et al. (2023, arXiv:2306.05685) judge-vs-human agreement,",
        "extended with chance-corrected Cohen's kappa and bootstrap 95% confidence intervals.",
        "",
        f"- **Comparisons:** {result.n_comparisons} "
        f"(from {result.n_human_votes} human votes, {result.n_gpt4_verdicts} GPT-4 verdicts)",
        f"- **Position-inconsistency rate** (GPT-4 verdict flipped between orderings): "
        f"{_fmt(result.position_inconsistency_rate)}",
        f"- **Human-human agreement ceiling** (Krippendorff's α over "
        f"{result.n_human_multi_rated} multiply-annotated comparisons): "
        f"{_fmt(result.human_krippendorff_alpha)}",
        "",
        "| Setup | n | Raw agreement | Cohen's κ | κ 95% CI |",
        "| --- | --- | --- | --- | --- |",
        _row("S1 (with ties)", result.with_ties),
        _row("S2 (ties excluded)", result.ties_excluded),
        "",
        "## By turn (with ties)",
        "",
        "| Turn | n | Raw agreement | Cohen's κ | κ 95% CI |",
        "| --- | --- | --- | --- | --- |",
        *(_row(f"Turn {turn}", result.by_turn[turn]) for turn in sorted(result.by_turn)),
        "",
        "## Notes",
        "",
        f"- Raw agreement overstates reliability: chance-corrected κ is markedly lower "
        f"(S2: {_fmt(result.ties_excluded.raw_agreement)} raw vs "
        f"{_fmt(result.ties_excluded.cohen_kappa)} κ). Report κ, not raw agreement.",
        "- Verdicts are aligned by model identity in a canonical ordering (order-invariant); "
        "human annotators are combined by majority vote; GPT-4 `tie (inconsistent)` counts as a "
        "tie and is surfaced as the position-inconsistency rate.",
        f"- The judge sits near the human ceiling: with ties, GPT-4↔human κ = "
        f"{_fmt(result.with_ties.cohen_kappa)} vs human↔human α = "
        f"{_fmt(result.human_krippendorff_alpha)} — GPT-4 agrees with humans about as well as "
        f"humans agree with each other. A judge is not more reliable than its reference.",
        "- S1 keeps ties; S2 excludes them (either side a tie), following the paper's two setups.",
        f"- Reproduce (keyless, no API calls): `judgelab report` — computed from the committed "
        f"CC-BY-4.0 snapshot with seed {REPORT_SEED} and {REPORT_RESAMPLES} bootstrap resamples. "
        f"CI runs `judgelab report --check` as a drift gate.",
        "",
    ]
    return "\n".join(lines)


def render_json(result: AgreementResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write_report() -> None:
    result = build_result()
    directory = reports_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MD_NAME).write_text(render_markdown(result), encoding="utf-8")
    (directory / JSON_NAME).write_text(render_json(result), encoding="utf-8")


def check_report() -> bool:
    """True iff the committed report matches a fresh recompute (the drift gate)."""
    result = build_result()
    directory = reports_dir()
    md_ok = (directory / MD_NAME).read_text(encoding="utf-8") == render_markdown(result)
    json_ok = (directory / JSON_NAME).read_text(encoding="utf-8") == render_json(result)
    return md_ok and json_ok


__all__ = [
    "JSON_NAME",
    "MD_NAME",
    "REPORT_RESAMPLES",
    "REPORT_SEED",
    "build_result",
    "check_report",
    "render_json",
    "render_markdown",
    "reports_dir",
    "write_report",
]
