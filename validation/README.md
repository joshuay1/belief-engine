# Paper results and analyses

Run verification and rebuild commands from the repository root, following
[REPRODUCING.md](../REPRODUCING.md).

## Included results

| Study | Saved outputs |
|---|---|
| Uptake and anchoring sweeps | [Trajectories, summaries, and figures](results/model_sensitivity_comparison/) |
| Main GPT-5.4-mini profile panel | [Austerity trajectories and provenance](camera_ready/profiles/) |
| Earlier five-topic baselines | [Three-trial comparison](results/all_topics_u0.2_a0.4_8models/) |
| Journalism external-judge check | [Scores and figure](results/journalism_u0.2_a0.4_8models/) |
| DEBATE replay | [Tables and figures](paper_artifacts/evidence_alignment/) |
| Group-held-out replay | [Participant predictions and selected parameters](results/parameter_recommendations_group_response5_evidence13x13_quality/) |
| Topic-held-out replay | [Participant predictions and selected parameters](results/parameter_recommendations_topic_response5_evidence13x13_quality/) |
| Individual replay examples | [Example trajectories](results/individual_opinion_tracking_response5_evidence13x13/) |
| Two-agent topic grids | [Trajectories and convergence summaries](paper_artifacts/two_agent_focus10/) |
| Original AQR extraction audit | [100-input extraction outputs](results/judgement_layer_aqr_llm/aqr_llm_gpt-4o-mini_20260426_141615/) |
| Earlier classifier summary | [Historical 1,000-input summary](results/judgement_layer_aqr/aqr_test_20260426_133422/) |

The [camera-ready packages](camera_ready/README.md) add the ten-topic stance
validation, common-evaluator comparison, endpoint solver, component study,
original 100-input AQR classifier audit, and supporting component checks.
The historical 1,000-input classifier summary is not used for the paper's retained
100-input correlation. Its exact selected rows are now in `camera_ready/aqr/`.

The original one-topic stance-calibration outputs are retained under
[`experiments/logs/`](../experiments/logs/).

## Code

- `scripts/`: experiment runners, plotting, and statistical summaries.
- `data/`: DEBATE loading and scale conversion.
- `fitting/` and `evaluation/`: parameter fitting and shared metrics.

Frozen outputs stay in `paper_artifacts/` and `results/`. Rebuilt outputs from
`make rebuild-replay-stats` go to `output/replay/` at the repository root.

The [final figure index](camera_ready/figures/README.md) identifies the exact
assets used in the camera-ready draft.
