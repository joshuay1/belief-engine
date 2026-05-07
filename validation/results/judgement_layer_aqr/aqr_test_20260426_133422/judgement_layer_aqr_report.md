# Judgement Layer Evaluation: Argument-Quality-Ranking-30k

This run evaluates the local argument-quality scorer against the
human/crowd-labelled Argument-Quality-Ranking-30k dataset.

## Inputs

- Data directory: `data/Argument-Quality-Ranking-30k`
- Split: `test`
- Sample size: `1000`
- Seed: `42`
- Classifier model: `models/argument_quality_classifier`

## Strength Reliability

| Target | N | Pearson r | Spearman rho | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|---:|---:|
| WA | 1000 | 0.5793 | 0.5248 | 0.1256 | 0.1746 | 0.0611 |
| MACE-P | 1000 | 0.5594 | 0.5561 | 0.3086 | 0.4200 | 0.2718 |
| quality_target | 1000 | 0.5873 | 0.5642 | 0.2052 | 0.2819 | 0.1664 |
| support_strength_target | 1000 | 0.5776 | 0.5633 | 0.2201 | 0.2999 | 0.1889 |

Primary target for paper reporting is `quality_target = mean(WA, MACE-P)`.
The local classifier was originally trained on this dataset family, so
these numbers validate in-domain scoring rather than out-of-domain
generalization.

## Sample Balance

| Stance | N |
|---:|---:|
| -1 | 500 |
| 1 | 500 |

- Unique topics in sample: `15`
- Full split rows available: `6315`

## Topic And Stance Stability

Topic metrics are computed for topics with at least `20` sampled arguments.

| Summary over topics | Pearson r | Spearman rho | MAE |
|---|---:|---:|---:|
| Mean | 0.5841 | 0.5348 | 0.2020 |
| Min | 0.4111 | 0.4133 | 0.3073 |
| Max | 0.7577 | 0.6432 | 0.0977 |

| Stance | N | Pearson r | Spearman rho | MAE |
|---:|---:|---:|---:|---:|
| -1 | 500 | 0.5978 | 0.5654 | 0.2076 |
| 1 | 500 | 0.5753 | 0.5629 | 0.2028 |

## Existing Extraction Cache Coverage

Existing DEBATE extraction caches do not currently cover AQR-30k rows.
A future extractor pass should use `sample_for_extractor.csv` as the
frozen benchmark input.

| Cache | Entries | Overlap | Non-empty overlap |
|---|---:|---:|---:|
| `extraction_cache.json` | 26882 | 0 | 0 |
| `extraction_cache_4o.json` | 121 | 0 | 0 |
| `extraction_cache_5.4-mini.json` | 121 | 0 | 0 |
| `extraction_cache_5.4-nano.json` | 121 | 0 | 0 |
| `extraction_cache_5.4.json` | 121 | 0 | 0 |
| `extraction_cache_claude-4-opus.json` | 121 | 0 | 0 |
| `extraction_cache_claude-4-sonnet.json` | 121 | 0 | 0 |
| `extraction_cache_claude-4.5-haiku.json` | 121 | 0 | 0 |
| `extraction_cache_claude-4.5-opus.json` | 121 | 0 | 0 |
| `extraction_cache_claude-4.5-sonnet.json` | 121 | 0 | 0 |
| `extraction_cache_deepseek-v3.2.json` | 121 | 0 | 0 |
| `extraction_cache_kimi-k2.json` | 121 | 0 | 0 |
| `extraction_cache_v1.json` | 27060 | 0 | 0 |

## Artifacts

- `predictions.csv`: sample rows plus classifier predictions.
- `sample_for_extractor.csv`: frozen sample for future LLM extraction/polarity evaluation.
- `metrics.json`: machine-readable summary metrics.
- `topic_metrics.csv`: strength reliability by topic.
- `stance_metrics.csv`: strength reliability by pro/con stance.
- `cache_coverage.csv`: extraction-cache overlap diagnostics.
- `figure_strength_reliability.pdf`: classifier strength versus human quality target.

## Interpretation

This validates the strength-scoring component against existing human
quality labels. It does not yet validate multi-claim extraction from
messy dialogue turns or binary-polarity failures on moderation cases.
Those should be handled by a smaller DEBATE-specific stress test.