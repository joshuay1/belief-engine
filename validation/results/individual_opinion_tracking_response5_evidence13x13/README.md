# Individual Opinion Tracking: response5 evidence13x13

Exploratory figure for mapping individual human opinion changes against the
Belief Engine trajectory implied by the response5 evidence-alignment
group-held-out parameter recommendation run.

## Sources

- Source: `validation/results/parameter_recommendations_group_response5_evidence13x13_quality/individual_results.csv`
- Plotting script: `validation/scripts/plot_top_matched_movers.py`

## Selection

Rows are selected from `preset_evidence_alignment`, restricted to
`preset_key == evidence_aligned_mover`, true movers whose predicted direction
matches the observed human direction. The plotting script reconstructs the
accepted-evidence trajectory for all matching candidates, scores each path by
direction reversals, opposing update steps, and excess path movement, then
keeps the top 20 back-and-forth trajectories with absolute opinion-change
error at or below 0.15.

The four-panel showcase then takes the four largest absolute human opinion
changes from that top-20 set.

The type showcase uses one individual trajectory example for each focused RMSE
surface group: all participants, evidence-aligned movers, evidence-opposed
movers, and stable participants. The current examples are selected by
reconstructed-path zigzag score, with `67d3cJDKBT` pinned as the
evidence-aligned example because it gives a visibly back-and-forth trajectory.

## Files

- `figure_top20_evidence_aligned_movers_tracking_group.pdf`: reconstructed model belief paths overlaid with observed human initial/final change.
- `figure_top4_largest_change_evidence_aligned_movers_row_group.pdf`: one-row showcase of the four largest human changes from the top-20 back-and-forth set.
- `figure_individual_type_showcase_group.pdf`: one-row conceptual showcase with one individual example from each evidence-alignment surface group.
- `top20_evidence_aligned_movers_group.csv`: selected rows, endpoint errors, and back-and-forth selection metrics.
- `top20_evidence_aligned_mover_trajectories_group.csv`: reconstructed accepted-evidence trajectory points.
- `top4_largest_change_evidence_aligned_movers_group.csv`: selected four-row subset used by the one-row showcase.
- `top4_largest_change_evidence_aligned_mover_trajectories_group.csv`: reconstructed accepted-evidence trajectory points for the four-row showcase.
- `type_showcase_examples_group.csv`: selected example rows for the four surface groups.
- `type_showcase_example_trajectories_group.csv`: reconstructed accepted-evidence trajectory points for the type showcase.

This folder intentionally excludes evidence-opposed movers so the examples are
about tracking human movement in the same direction as accepted evidence.
