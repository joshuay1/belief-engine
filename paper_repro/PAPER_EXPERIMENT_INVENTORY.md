> Historical reviewer snapshot documentation (May 2026). For the current code candidate, see [CAMERA_READY.md](CAMERA_READY.md). The later result packages are not yet included.

# Paper Experiment Inventory

Checked against:

- Active manuscript: paper manuscript source, not included in reviewer code artifact.
- Older manuscript not treated as current: older draft, not included in reviewer code artifact.

This inventory defines the experiment-level reproducibility boundary for the
current paper. The goal is to make every quantitative figure, table, and scalar
claim traceable to a source run, script, and release artifact.

## Main-Paper Experiments

| Paper object | What must be reproducible | Current source in this repo | Notes |
|---|---|---|---|
| Internal-to-external stance calibration, reported as `r = 0.967` | Controlled stance sweep from `S=-1` to `S=1`, judged externally | `experiments/logs/stance_correlation_20260213_165900/data/results.csv`; config `config/selected_results/canonical_stance_correlation.yaml` | Recomputes `subject` Pearson `r=0.967386`, slope `0.863266`; `moltbot` comparison is `r=0.949793`. This is a scalar paper claim, not currently a main figure. |
| Figure `sweep_be_agent.pdf` | BE-agent parameter sweeps over evidence uptake `u` and prior anchoring `a` across GPT-4o-mini, Qwen3.5:9b, and Gemma4:E4b | `validation/results/model_sensitivity_comparison/figure_model_trajectory_sweep_be_agent.pdf`; raw logs under `logs/*canonical_sensitivity_sweep_deterministic_counter_*`; script `validation/scripts/plot_model_sensitivity_comparison.py` | The paper copy is byte-identical to this PDF. |
| Figure `configure.pdf` | Illustrative two-agent Open-minded/Stubborn profile panel | Reviewer artifact includes the compact two-agent bundle at `validation/paper_artifacts/two_agent_focus10/topic_grids/` | This is an illustrative example panel. It should not be treated as a core exact-reproduction claim unless the final manuscript keeps it as a cited quantitative artifact. |
| Figure `atheism.pdf` | Matched atheism prompt-baseline comparison: four BE variants plus prompt self-update and RAG + self-update | `validation/results/all_topics_u0.2_a0.4_8models/atheism/figures/topic_grids_6panel/atheism_6panel_gpt54mini_baselines_grid_3x2.pdf`; script `validation/scripts/build_6panel_topic_grids_and_metrics.py` | Paper copy is byte-identical. This is also the source family for the five-topic prompt-based table. |
| Figure `parameter_alignment.pdf` | DEBATE replay RMSE surfaces for all participants, evidence-aligned movers, evidence-opposed movers, and stable participants | `validation/paper_artifacts/evidence_alignment/figure_parameter_rmse_surfaces_evidence_alignment_focus.pdf`; script family `validation/scripts/run_parameter_recommendations.py` and `validation/scripts/generate_parameter_recommendation_figures.py` | The reviewer artifact keeps the paper-facing copy only; duplicated source-result copies were pruned. |
| Figure `example.pdf` | Individual DEBATE replay examples for the four evidence-alignment response patterns | `validation/results/individual_opinion_tracking_response5_evidence13x13/figure_individual_type_showcase_group.pdf`; script `validation/scripts/plot_top_matched_movers.py` | Paper copy is byte-identical. This is a visualization derived from the DEBATE replay results, not a separate model fit. |

## Appendix And Supporting Experiments

| Paper object | What must be reproducible | Current source in this repo | Notes |
|---|---|---|---|
| Table `five-topic-prompt-based-check` | Five-topic summary over Social media, Journalism, Compulsory voting, Atheism, and Libertarianism | `validation/results/all_topics_u0.2_a0.4_8models/metrics/all_topics_6panel_metrics.csv` and `.tex`; script `validation/scripts/build_6panel_topic_grids_and_metrics.py` | Same measurement caveat as the main atheism panel: BE rows use internal stance, baselines use an external judge. |
| Table `aqr-judgement` | AQR 100-row extraction/polarity check and strength-reliability comparison | `validation/results/judgement_layer_aqr/aqr_test_20260426_133422/`; `validation/results/judgement_layer_aqr_llm/aqr_llm_gpt-4o-mini_20260426_141615/`; scripts `validation/scripts/evaluate_judgement_layer_aqr.py` and `validation/scripts/run_aqr_llm_extraction_eval.py` | The LLM extraction run gives extraction rate `0.970`, strongest-claim polarity accuracy `0.969`, and LLM-strength/quality `r=0.089`; the classifier report gives classifier-strength/quality `r=0.720` in the paper text. |
| Tables `evidence-alignment-baselines`, `parameter-importance`, `parameter-importance-movers` | DEBATE held-out RMSE and parameter-importance decompositions | `validation/paper_artifacts/evidence_alignment/*.csv` | These are derived from the same DEBATE replay run as `parameter_alignment.pdf`. |
| Appendix/robustness table `debate-rmse-uncertainty` | Fold-level confidence intervals and paired tests for DEBATE RMSE | `validation/paper_artifacts/evidence_alignment/debate_rmse_confidence_intervals.csv`; `validation/paper_artifacts/evidence_alignment/debate_rmse_fold_tests.csv`; script `validation/scripts/summarize_debate_rmse_uncertainty.py` | Tests compare held-out fold RMSE values within the same split/fold. With five folds, treat p-values as robustness checks, not a replacement for participant-level bootstrap. |
| Figure `multi-agent.png` and stability tables | Symmetrical six-condition trajectory stability benchmark | `experiments/selected_results/canonical_6cond_symm/`; paper figure matches `experiments/selected_results/canonical_6cond_symm/topic_05.png` | Covers 2 architectures x 3 update conditions x 10 trials; table rows use per-agent metrics (`n=20` per condition). |
| Ten-topic two-agent topic grids and convergence table | Generated two-agent Open/Committed topic grids over 10 topics, 4 profile pairings, 3 trials per cell | `validation/paper_artifacts/two_agent_focus10/` | The paper topic-grid PDFs are the frozen `gpt-4o-mini` focus-10 artifact bundle, not the later `gpt-5.4-mini` focus-10 folder. |

## Presentation Or Non-Experiment Objects

These should be kept in the paper artifact bundle, but they do not require a
separate empirical rerun:

- `diagram.png`: architecture diagram, if included on the manuscript side.
- Design-rationale table.
- Preset-definition table.
- Hyperparameter table.

## Currently Excluded From The Boundary

- `submission_without_comments.tex` and its older figure set.
- Figures in the active manuscript inside `\iffalse`, especially
  `parameter_topic.pdf`, unless the section is re-enabled.
- Exploratory validation folders not referenced by the active paper.
- Large raw/intermediate folders unless selected as canonical sources above.

## Immediate Cleanup Recommendation

The next cleanup step is not to fix exploratory figure choices, but to freeze the
canonical release boundary:

1. Keep the experiment groups listed above as the paper-facing scope.
2. Move release work onto a branch such as `paper-repro`.
3. Build an allowlist of source files, configs, scripts, small tables, and
   selected artifacts needed to verify those groups.
4. Put large raw/intermediate runs in an external archive with checksums.
