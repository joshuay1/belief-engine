# LLM Judgement Layer Evaluation: AQR-30k

This run calls the real `ArgumentExtractionAgent` on the frozen
Argument-Quality-Ranking-30k sample and compares outputs to human
stance and quality labels.

## Inputs

- Sample: `validation/results/judgement_layer_aqr/aqr_test_20260426_133422/sample_for_extractor.csv`
- Backend: `openai`
- Model: `gpt-4o-mini`
- Rows evaluated in this run: `100`
- Cache path: `validation/data/cache/extractions/aqr/aqr_extraction_cache_gpt-4o-mini.json`

## Extraction And Polarity

| Metric | Value |
|---|---:|
| Extraction rate (`n_extracted > 0`) | 0.970 |
| Exact-one extraction rate | 0.770 |
| Over-split rate (`n_extracted > 1`) | 0.200 |
| Mean extracted claims | 1.19 |

## Polarity Recovery

| Polarity rule | N | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Aggregate signed pressure | 96 | 0.969 | 0.969 | 0.968 | 0.969 |
| Strongest extracted claim | 97 | 0.969 | 0.970 | 0.968 | 0.969 |

## Strength Reliability

| Target | N | Pearson r | Spearman rho | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|---:|---:|
| WA | 97 | 0.1184 | 0.1091 | 0.2840 | 0.3245 | -0.2368 |
| MACE-P | 97 | 0.0678 | 0.0884 | 0.3362 | 0.3811 | -0.0291 |
| quality_target | 97 | 0.0892 | 0.1062 | 0.2754 | 0.3181 | -0.1330 |

## Artifacts

- `llm_extraction_predictions.csv`: row-level predictions and extracted claims.
- `llm_metrics.json`: machine-readable metric summary.
- `figure_llm_strength_reliability.pdf`: LLM strength versus human quality.

## Notes

AQR-30k rows are already single human-written arguments, so exact-one
extraction is desirable. DEBATE dialogue still needs a separate stress
test for multi-claim messages and moderation/exception language.