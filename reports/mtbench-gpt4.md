# MT-Bench — GPT-4 judge vs human agreement

Reproduction of Zheng et al. (2023, arXiv:2306.05685) judge-vs-human agreement,
extended with chance-corrected Cohen's kappa and bootstrap 95% confidence intervals.

- **Comparisons:** 1814 (from 3355 human votes, 2400 GPT-4 verdicts)
- **Position-inconsistency rate** (GPT-4 verdict flipped between orderings): 0.160
- **Human-human agreement ceiling** (Krippendorff's α over 961 multiply-annotated comparisons): 0.485

| Setup | n | Raw agreement | Cohen's κ | κ 95% CI |
| --- | --- | --- | --- | --- |
| S1 (with ties) | 1814 | 0.672 | 0.505 | [0.472, 0.538] |
| S2 (ties excluded) | 1078 | 0.884 | 0.767 | [0.726, 0.804] |

## By turn (with ties)

| Turn | n | Raw agreement | Cohen's κ | κ 95% CI |
| --- | --- | --- | --- | --- |
| Turn 1 | 910 | 0.678 | 0.516 | [0.467, 0.561] |
| Turn 2 | 904 | 0.666 | 0.492 | [0.446, 0.539] |

## Notes

- Raw agreement overstates reliability: chance-corrected κ is markedly lower (S2: 0.884 raw vs 0.767 κ). Report κ, not raw agreement.
- Verdicts are aligned by model identity in a canonical ordering (order-invariant); human annotators are combined by majority vote; GPT-4 `tie (inconsistent)` counts as a tie and is surfaced as the position-inconsistency rate.
- The judge sits near the human ceiling: with ties, GPT-4↔human κ = 0.505 vs human↔human α = 0.485 — GPT-4 agrees with humans about as well as humans agree with each other. A judge is not more reliable than its reference.
- S1 keeps ties; S2 excludes them (either side a tie), following the paper's two setups.
- Reproduce (keyless, no API calls): `judgelab report` — computed from the committed CC-BY-4.0 snapshot with seed 0 and 2000 bootstrap resamples. CI runs `judgelab report --check` as a drift gate.
