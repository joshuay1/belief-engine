# Main two-agent profile panel

These twelve austerity debates are the GPT-5.4-mini campaign used for the main
profile panel (`configure.pdf`). There are three trials for each of four pairings,
with 15 rounds and initial stances +0.75 and -0.75. The figure shows Open/Open,
Committed/Open, and Committed/Committed. The saved label `Committed` is displayed
as Stubborn in the paper. Profiles use (uptake, anchoring) = (0.40, 0.20) and
(0.10, 0.80), respectively.

`summary.csv` and `stance_trajectories.csv` contain only this topic. The archive's
local raw-directory column was removed. `provenance.json` records the original
file hashes and the plotting entry point. Its plotting source is unchanged from
the archived campaign. The original function `generate_figures` accepts these
CSV tables as lists of row dictionaries with preset `human_showcase_focus10`.
The final PDF is included, so rebuilding its precise layout is not necessary to
check the values. Run `make check-figures` to verify coverage and endpoints.
