# DEBATE RMSE Fold-Level Uncertainty

Computed from the canonical quality-filtered DEBATE parameter recommendation
`individual_results.csv` files. Each model has one held-out RMSE per
outer fold for the group-held-out split and one per outer fold for the
topic-held-out split.

Confidence intervals are t intervals over the five outer folds. Paired
tests compare fold RMSE values within the same split and fold. With only
five folds, these tests should be read as fold-level robustness checks,
not as a substitute for a participant-level bootstrap or a new rerun.

Negative RMSE deltas mean the first model has lower RMSE than the
reference model.

## Data-Driven Reading

Generated from the confidence-interval and paired-test tables above:

- `group-held-out`: `Evidence-alignment parameters` has aggregate RMSE `0.3657` and fold-mean RMSE `0.3652` (95% fold CI `[0.3373, 0.3930]`).
- `group-held-out` vs `Prior only`: mean fold RMSE delta `-0.1231` (95% delta CI `[-0.1446, -0.1016]`), with lower RMSE in `5/5` folds.
- `group-held-out` vs `Global parameters`: mean fold RMSE delta `-0.0998` (95% delta CI `[-0.1123, -0.0873]`), with lower RMSE in `5/5` folds.
- `group-held-out`: the lowest aggregate RMSE row is `Response-type diagnostic` at `0.2824`. Treat this as diagnostic because response-type presets are based on observed outcome classes rather than pre-debate deployable groups.
- `topic-held-out`: `Evidence-alignment parameters` has aggregate RMSE `0.3652` and fold-mean RMSE `0.3663` (95% fold CI `[0.3563, 0.3762]`).
- `topic-held-out` vs `Prior only`: mean fold RMSE delta `-0.1220` (95% delta CI `[-0.1523, -0.0917]`), with lower RMSE in `5/5` folds.
- `topic-held-out` vs `Global parameters`: mean fold RMSE delta `-0.0994` (95% delta CI `[-0.1175, -0.0813]`), with lower RMSE in `5/5` folds.
- `topic-held-out`: the lowest aggregate RMSE row is `Response-type diagnostic` at `0.2826`. Treat this as diagnostic because response-type presets are based on observed outcome classes rather than pre-debate deployable groups.

The paired t-test columns quantify the fold-level mean deltas. The Wilcoxon
column is intentionally retained as a small-sample sign/rank check; with
five folds, all folds moving in the same direction can still yield a
two-sided Wilcoxon p-value of `0.0625`.

## Fold-Level RMSE Confidence Intervals

| Split | Model | Aggregate RMSE | Fold mean RMSE | 95% fold CI | Folds |
| --- | --- | --- | --- | --- | --- |
| group-held-out | Prior only | 0.4891 | 0.4883 | [0.4502, 0.5264] | 5 |
| group-held-out | Global parameters | 0.4657 | 0.4650 | [0.4317, 0.4984] | 5 |
| group-held-out | Participant-type parameters | 0.4663 | 0.4658 | [0.4349, 0.4967] | 5 |
| group-held-out | Topic-category parameters | 0.4660 | 0.4654 | [0.4327, 0.4980] | 5 |
| group-held-out | Evidence-alignment parameters | 0.3657 | 0.3652 | [0.3373, 0.3930] | 5 |
| group-held-out | Response-type diagnostic | 0.2824 | 0.2818 | [0.2546, 0.3090] | 5 |
| topic-held-out | Prior only | 0.4891 | 0.4883 | [0.4541, 0.5225] | 5 |
| topic-held-out | Global parameters | 0.4659 | 0.4657 | [0.4439, 0.4874] | 5 |
| topic-held-out | Participant-type parameters | 0.4674 | 0.4668 | [0.4434, 0.4903] | 5 |
| topic-held-out | Topic-category parameters | 0.4670 | 0.4667 | [0.4462, 0.4871] | 5 |
| topic-held-out | Evidence-alignment parameters | 0.3652 | 0.3663 | [0.3563, 0.3762] | 5 |
| topic-held-out | Response-type diagnostic | 0.2826 | 0.2826 | [0.2636, 0.3017] | 5 |

## Primary Fold-Level Tests: Evidence-alignment parameters

| Split | Comparison | Mean RMSE delta | 95% delta CI | paired t p | Wilcoxon p | Lower-RMSE folds |
| --- | --- | --- | --- | --- | --- | --- |
| group-held-out | Evidence-alignment parameters vs Prior only | -0.1231 | [-0.1446, -0.1016] | 9.11e-05 | 0.0625 | 5/5 |
| group-held-out | Evidence-alignment parameters vs Global parameters | -0.0998 | [-0.1123, -0.0873] | 2.44e-05 | 0.0625 | 5/5 |
| topic-held-out | Evidence-alignment parameters vs Prior only | -0.1220 | [-0.1523, -0.0917] | 3.64e-04 | 0.0625 | 5/5 |
| topic-held-out | Evidence-alignment parameters vs Global parameters | -0.0994 | [-0.1175, -0.0813] | 1.07e-04 | 0.0625 | 5/5 |

## Files

- `debate_rmse_fold_metrics.csv`
- `debate_rmse_confidence_intervals.csv`
- `debate_rmse_fold_tests.csv`
