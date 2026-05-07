# Parameter Recommendation Report

- Run ID: `parameter_recommendations_topic_20260428_132228`
- Split: `topic`
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
| preset_evidence_alignment |       2495 |     0.202401 |      0.365247 |           0.691968 |                    0.908617 |         0.060597    |
| preset_global             |       2495 |     0.352179 |      0.465903 |           0.310342 |                    0.32024  |         0.105509    |
| preset_participant_type   |       2495 |     0.3489   |      0.467437 |           0.297217 |                    0.306212 |         0.0975441   |
| preset_response_type      |       2495 |     0.148703 |      0.282619 |           0.899375 |                    0.99479  |         0.0485269   |
| preset_topic_category     |       2495 |     0.356552 |      0.466966 |           0.304964 |                    0.30982  |         0.108042    |
| prior_only_no_change      |       2495 |     0.282958 |      0.489074 |         nan        |                    0.552705 |         0.000695035 |

## Recommended Presets

| scheme             | preset_label            |   recommended_update_s |   recommended_anchor_s |   heldout_rmse |   heldout_pearson_r |   heldout_direction_accuracy | recommended_use                                                                                         |
|:-------------------|:------------------------|-----------------------:|-----------------------:|---------------:|--------------------:|-----------------------------:|:--------------------------------------------------------------------------------------------------------|
| evidence_alignment | Evidence-aligned movers |                  0.4   |                   0.5  |    0.398289    |            0.69109  |                     0.96679  | Diagnostic group for movers whose accepted evidence pressure points in their movement direction.        |
| evidence_alignment | All participants        |                  0.035 |                   0.6  |  nan           |          nan        |                   nan        | Diagnostic evidence-alignment reference panel for all participants.                                     |
| evidence_alignment | Evidence-opposed movers |                  0.005 |                   0.3  |    0.659195    |            0.584932 |                     0.6325   | Diagnostic group for movers whose accepted evidence pressure points opposite their movement.            |
| evidence_alignment | Stable despite pressure |                  0.005 |                   1    |    0.00688734  |          nan        |                     1        | Diagnostic group for stable participants despite non-trivial accepted evidence pressure.                |
| evidence_alignment | Stable low-pressure     |                  0.005 |                   1    |    0.000867021 |          nan        |                     1        | Diagnostic group for stable participants with weak or mixed accepted evidence pressure.                 |
| evidence_alignment | Weak-signal movers      |                  0.005 |                   0.2  |    0.6477      |            0.59736  |                     0.637931 | Diagnostic group for movers with weak or mixed accepted evidence pressure.                              |
| global             | General default         |                  0.035 |                   0.6  |    0.465903    |            0.310342 |                     0.32024  | Default when no pre-debate type is available.                                                           |
| participant_type   | Committed               |                  0.075 |                   0.6  |    0.376787    |            0.389254 |                     0.225758 | Use for agents with strong consistent priors and high inertia.                                          |
| participant_type   | Leaning                 |                  0.035 |                   0.8  |    0.464784    |            0.221784 |                     0.336245 | Use for agents with moderate but consistent priors.                                                     |
| participant_type   | Open / conflicted       |                  0.02  |                   0.6  |    0.513784    |            0.262562 |                     0.334495 | Use for agents with weak, neutral, or internally conflicted priors.                                     |
| response_type      | Boundary switchers      |                  0.02  |                   0.02 |    0.251889    |            0.996355 |                     1        | Diagnostic group for participants who crossed the agree/disagree boundary by one Likert step.           |
| response_type      | Reinforced              |                  0.05  |                   1.5  |    0.291787    |            0.841255 |                     0.971302 | Diagnostic group for participants who moved further toward their initial side.                          |
| response_type      | Softened                |                  0.005 |                   0.4  |    0.146791    |            0.943993 |                     1        | Diagnostic group for participants who moved toward the agree/disagree boundary without crossing it.     |
| response_type      | Stable                  |                  0.005 |                   1    |    0.00630213  |          nan        |                     1        | Diagnostic group for participants with no Likert movement.                                              |
| response_type      | Strong switchers        |                  0.005 |                   0.02 |    0.637494    |            0.943913 |                     1        | Diagnostic group for participants who crossed the agree/disagree boundary by more than one Likert step. |
| topic_category     | Broad social/political  |                  0.02  |                   0.6  |    0.462346    |            0.30798  |                     0.308983 | Use for broad social or political propositions.                                                         |
| topic_category     | Technical/factual       |                  0.05  |                   0.6  |    0.486421    |            0.294604 |                     0.313433 | Use for more technical or factual propositions.                                                         |

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