# Parameter Recommendation Report

- Run ID: `parameter_recommendations_group_20260428_131334`
- Split: `group`
- Clean mode: `quality`
- Exposure mode: `received`
- Uptake mode: `human_like`
- Strength source: `classifier`
- Participants: `2495`
- Folds: `5`
- Objective: `rmse`

## Interpretation

This run selects `update_s` and `anchor_s` on train folds only, then
evaluates the selected presets on held-out participants. Participant
types are assigned from pre-debate prior extremity and initial-text
consistency, so type-specific presets are deployable before a debate
starts.

## Held-Out Scheme Metrics

| model                     |   change_n |   change_mae |   change_rmse |   change_pearson_r |   change_direction_accuracy |   convergence_error |
|:--------------------------|-----------:|-------------:|--------------:|-------------------:|----------------------------:|--------------------:|
| preset_evidence_alignment |       2495 |     0.202629 |      0.365691 |           0.693101 |                    0.907816 |         0.0601335   |
| preset_global             |       2495 |     0.3521   |      0.465681 |           0.312098 |                    0.317836 |         0.105556    |
| preset_participant_type   |       2495 |     0.344622 |      0.466341 |           0.304148 |                    0.315832 |         0.0946359   |
| preset_response_type      |       2495 |     0.148623 |      0.282439 |           0.897918 |                    0.99479  |         0.0483307   |
| preset_topic_category     |       2495 |     0.352807 |      0.465996 |           0.309657 |                    0.31503  |         0.105275    |
| prior_only_no_change      |       2495 |     0.282958 |      0.489074 |         nan        |                    0.552705 |         0.000695035 |

## Recommended Presets

| scheme             | preset_label            |   recommended_update_s |   recommended_anchor_s |   heldout_rmse |   heldout_pearson_r |   heldout_direction_accuracy | recommended_use                                                                                         |
|:-------------------|:------------------------|-----------------------:|-----------------------:|---------------:|--------------------:|-----------------------------:|:--------------------------------------------------------------------------------------------------------|
| evidence_alignment | Evidence-aligned movers |                  0.4   |                   0.5  |    0.394781    |            0.700684 |                     0.9631   | Diagnostic group for movers whose accepted evidence pressure points in their movement direction.        |
| evidence_alignment | All participants        |                  0.035 |                   0.6  |  nan           |          nan        |                   nan        | Diagnostic evidence-alignment reference panel for all participants.                                     |
| evidence_alignment | Evidence-opposed movers |                  0.005 |                   0.3  |    0.661591    |            0.576174 |                     0.6325   | Diagnostic group for movers whose accepted evidence pressure points opposite their movement.            |
| evidence_alignment | Stable despite pressure |                  0.005 |                   1    |    0.00688734  |          nan        |                     1        | Diagnostic group for stable participants despite non-trivial accepted evidence pressure.                |
| evidence_alignment | Stable low-pressure     |                  0.005 |                   1    |    0.000867021 |          nan        |                     1        | Diagnostic group for stable participants with weak or mixed accepted evidence pressure.                 |
| evidence_alignment | Weak-signal movers      |                  0.005 |                   0.2  |    0.652353    |            0.586854 |                     0.637931 | Diagnostic group for movers with weak or mixed accepted evidence pressure.                              |
| global             | General default         |                  0.035 |                   0.6  |    0.465681    |            0.312098 |                     0.317836 | Default when no pre-debate type is available.                                                           |
| participant_type   | Committed               |                  0.075 |                   0.6  |    0.37554     |            0.394899 |                     0.221212 | Use for agents with strong consistent priors and high inertia.                                          |
| participant_type   | Leaning                 |                  0.05  |                   0.8  |    0.460877    |            0.236373 |                     0.372635 | Use for agents with moderate but consistent priors.                                                     |
| participant_type   | Open / conflicted       |                  0.02  |                   0.6  |    0.514252    |            0.263019 |                     0.336237 | Use for agents with weak, neutral, or internally conflicted priors.                                     |
| response_type      | Boundary switchers      |                  0.02  |                   0.02 |    0.251227    |            0.996121 |                     1        | Diagnostic group for participants who crossed the agree/disagree boundary by one Likert step.           |
| response_type      | Reinforced              |                  0.05  |                   1.5  |    0.291787    |            0.841255 |                     0.971302 | Diagnostic group for participants who moved further toward their initial side.                          |
| response_type      | Softened                |                  0.005 |                   0.4  |    0.147067    |            0.942882 |                     1        | Diagnostic group for participants who moved toward the agree/disagree boundary without crossing it.     |
| response_type      | Stable                  |                  0.005 |                   1    |    0.00630213  |          nan        |                     1        | Diagnostic group for participants with no Likert movement.                                              |
| response_type      | Strong switchers        |                  0.005 |                   0.02 |    0.63699     |            0.943938 |                     1        | Diagnostic group for participants who crossed the agree/disagree boundary by more than one Likert step. |
| topic_category     | Broad social/political  |                  0.02  |                   0.6  |    0.462299    |            0.310384 |                     0.310958 | Use for broad social or political propositions.                                                         |
| topic_category     | Technical/factual       |                  0.05  |                   0.6  |    0.48164     |            0.305622 |                     0.332623 | Use for more technical or factual propositions.                                                         |

## Files

- `parameter_recommendation_summary.json`
- `individual_results.csv`
- `heldout_metrics.csv`
- `stratified_metrics.csv`
- `group_results.csv`
- `fold_selected_params.csv`
- `parameter_recommendations.csv`
- `figure_parameter_presets.pdf`
- `figure_global_vs_type_specific.pdf`