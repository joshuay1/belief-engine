# Camera-ready results

These compact packages support the new empirical results in the final draft.
Run from the repository root after `pip install -e '.[analysis]'`:

```bash
make rebuild-paper
```

The command uses saved inputs, makes no model calls, and writes to `output/paper/`.
It fails if a checked result disagrees with the recorded analysis. It can take a
few minutes. To run one study:

```bash
python -B validation/scripts/paper/rebuild.py --study baselines
```

| `--study` | Inputs | Paper result checked |
|---|---|---|
| `stance` | `stance_alignment/results.csv`, frozen configuration and prompt-bearing sources | 630 responses, 210 cell means, correlation .959, slope .928, topic-clustered intervals and per-topic estimates |
| `baselines` | `baselines/runs.csv`, `trajectories.csv`, protocol and historical prompts | 150 debates, final gaps .5282 / 1.1158 / 1.0182, condition and simultaneous contrast intervals, topic results, leave-one-topic-out and sign-flip checks |
| `endpoints` | Shared numerical evidence and predictions in `replay/`, plus `endpoints/spec.yaml` | 2,441 eligible records, 958 matching endpoints, 869 earlier gaps ≥ .20, clustered intervals and the two selected trajectories |
| `components` | `components/cells.csv`, condition definitions and protocol | 500 debates, five final-gap contrasts and adjusted intervals, 6.34 claims per message versus one whole-message record |
| `aqr` | `aqr/selected_100.csv` | Exact 100-input classifier correlation .720, 97 successful extractions, 94 correct polarities and LLM-strength correlation .089 |
| `checks` | Fixed-stream replay predictions, natural-dedup predictions, controlled duplicate fixture | Reported scoring/updater contrasts, natural-dedup loss differences and repeated-claim invariance |

`stance` also regenerates the ten-topic alignment plots. `endpoints` plots the
same selected trajectories with a compact layout. Their plotted values reproduce
the paper; typography need not be byte-identical to the manuscript artwork.
The other studies write numerical tables used in the corresponding prose/tables.

## What is preserved

- `expected/` folders contain original numerical outputs used for comparisons.
- `provenance.json` records source hashes, file projections and code adaptations.
- Original numerical routines are kept in `../scripts/paper/frozen/`. The public
  wrapper calls their offline analysis functions. Historical live-run entry points
  inside these files are not the reproduction interface.
- Baseline generated dialogues accompany the judged trajectories. Component
  outcomes retain all 500 conditions/blocks and numeric manipulation counts,
  without the much larger per-turn memory and debugging dumps. `source_runs.json`
  records the hashes of all 500 original result files checked during packaging.
- The AQR table contains only the exact 100 selected inputs, identified by text
  hashes and row positions, plus labels and predictions. It omits original dataset
  text. These are the first 100 rows of the saved, seed-42 balanced classifier audit,
  matched in order to the extraction audit. No performance-based reselection was
  performed. Classifier correlation uses all 100 rows; LLM correlation uses the 97
  inputs that yielded a claim. The target is the mean of WA and MACE-P labels.
- Replay inputs contain numerical predictions, strengths, signs and anonymized
  identifiers, not human transcripts. Dataset access remains separate.

## Verification scope

The stance, baseline, endpoint and component rebuilds check their saved primary
confidence intervals as well as point estimates. The endpoint calculation starts
from the numerical evidence stream and solves for matching endpoints again.
Component contrasts start from individual debate outcomes, not saved contrasts.

For the earlier scoring/updater replay, sorting the released hashed cluster IDs
changes which groups receive a given bootstrap draw. The rebuild checks point
estimates exactly and the uncertainty conclusions reported in the paper. The
original intervals remain in `replay/expected/paired_cluster_bootstrap.csv`.
Natural-deduplication checks recompute the reported loss differences from saved
out-of-fold predictions; their original intervals are supplied for reference.
The controlled fixture checks its recorded active-evidence arithmetic, rather
than claiming a new audit of natural-language similarity detection.

This is saved-output reproduction. Live experiments additionally require dataset
access, the specified model services, and the original run configuration and
prompts. They can produce different language-model outputs. The published scorer
and its pinned revision are described in the root reproduction guide.
