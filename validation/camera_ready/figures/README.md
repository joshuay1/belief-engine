# Final manuscript figures

`index.json` maps all 17 figure assets to exact release files and SHA-256 hashes.
Files already present in the frozen bundles are referenced rather than duplicated.
This folder holds only the remaining final versions, including the architecture
PDF and the ten-topic stance-alignment TikZ source. Numerical rebuilds may use a
different layout from these final manuscript assets.

| Figure content | Numerical inputs / analysis |
|---|---|
| Architecture | Schematic; final `figure1_s.pdf` here |
| Three-model sweeps | `validation/results/model_sensitivity_comparison/` |
| Main profile panel | `validation/camera_ready/profiles/` (GPT-5.4-mini, austerity) |
| Earlier atheism / journalism comparisons | `validation/results/all_topics_u0.2_a0.4_8models/` and `validation/results/journalism_u0.2_a0.4_8models/` |
| Replay heatmap | `validation/paper_artifacts/evidence_alignment/` |
| Participant examples | `validation/results/individual_opinion_tracking_response5_evidence13x13/` |
| Endpoint paths / stance alignment | `make rebuild-paper` |
| Appendix topic grids | `validation/paper_artifacts/two_agent_focus10/` |

Run `make check-figures` from the repository root. It checks all file hashes and
reconciles the 150 BE sweep runs and 12 profile runs with their round-level data.
The sweep curves retain the historical incremental update semantics. Their
configured seed limit was ten, with eight admitted seeds in the source audit.
The main profile panel is a different campaign from the appendix topic grids.
No new experiments or replacement result values were introduced.
