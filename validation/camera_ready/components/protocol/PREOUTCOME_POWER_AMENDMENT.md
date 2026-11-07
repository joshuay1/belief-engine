# Pre-outcome power, measurement, and execution amendments

## Status and chronology

This is a single, outcome-blind amendment to
`neurips_2026_component_necessity_factorial_v1`. It was made on 2026-07-25
after the original 5-topic × 3-trial registration and dry run, but before any
response, extraction, external-judge score, or scientific outcome from a
component-factorial cell existed. The original dry run contains 135
configuration files and zero `results.json` files.

The amendment was triggered by three checks performed without reading any
component-factorial outcome:

1. A variance-based power/MDE audit of completed historical campaigns showed
   that 5×3 was a large-effect screen, not a reviewer-robust design for modest
   component effects.
2. A literal audit of the reviewer request showed that the original four
   contrasts did not isolate evidence scoring.
3. A later skeptical-review audit, still before any paid component-factorial
   call or scientific outcome, showed that the deduplication contrast also
   needed a realized-exposure check. Otherwise, a null D1-versus-D0 result
   could simply mean that no natural duplicate triggered the gate.
4. A final outcome-blind execution audit found that generic campaign-level
   parallelism did not enforce a complete topic-by-trial barrier and left
   condition submission position fixed. It also found that the three-cell
   canary was not a complete paired unit.
5. An independent outcome-blind audit of the frozen block scheduler then
   identified two narrow crash states not exercised by the original resume
   fixture: stale `running` attempt/invocation records after process
   termination, and an atomic `results.json` marker written before its parent
   ledger was flushed.

No outcome can control continuation. The only permitted stops remain functional
contract failure, credential/provider failure, disk-safety failure, or an
explicit operator stop.

The third check changed telemetry and analysis outputs only. It did not change
the 500 cells, ten conditions, primary or secondary endpoints, five-contrast
family, uncertainty, continuation rule, or 54,000-call budget.

The fourth check changed only execution order, execution provenance, and the
canary boundary. It did not change the 500 cells or registered inference. The
canary increased from three cells (330 successful calls) to the first complete
ten-condition block (1,080 successful calls); those ten cells were already
part of the unconditional 54,000-call campaign.

The fifth check changed only recovery of scheduling provenance. It did not
change any cell, factor, endpoint, contrast, multiplicity or continuation
rule, or the call budget. The reconciliation reads a prior frozen functional
checkpoint and the separate execution sidecars, hashes but never parses
`results.json`, and never modifies completed raw-cell bytes.

## Preserved original registration

| Original artifact | SHA-256 |
|---|---|
| Specification | `e87b11177c0604ff25021f6daeb3b8f26fe1585a0ecacacf217453b9a3c8060d` |
| Comparison matrix | `d3c37739001105796869508fbee97514028758417fccd2613c6d4bf1785d37c7` |
| Registry (135 cells) | `b497f09a0ca7704f1686bf3494d6f88e466cc95f756a996a4fe4a5fb65416939` |
| Design inventory | `2b5e66bbb4ca6b0c6ec2191befcb35d31a70946cd28af5e3cb306a32b547a582` |
| Implementation lock | `9213d15b745a385b67cb1573fe1bb81aaad34cc0d592cfe490b1a6dc56aedff5` |
| Original dry-run manifest | `f5815c986981daf78c5d91eee8ac7951a76fb6dd93fee6ca9159990d0f2846da` |
| Original preflight verification | `2d6760b279ee04022668209b2c64bebccadef17a65a5ef51f93a9c147d36e519` |

The implementation commit in the original lock remains
`32b1f976769628b3beccd8b477a68bccbb8b9b03`, and the last pre-amendment
repository commit was `4895ec2`. The original design was 15 paired blocks,
135 debates, and 14,400 successful provider calls excluding retries.

## Amended design

The amended design uses the fixed ten-topic benchmark list and five trials per
topic. Each of 50 paired topic-by-trial blocks contains:

- the full 2×2×2 structured-representation × response-retrieval × persistence
  core with deduplication on;
- the targeted deduplication-off arm at the full setting; and
- a targeted scale-matched uniform-strength arm at the full setting.

This yields 10 conditions per block, 500 debates, 300 structured debates, 200
flat debates, and 54,000 successful provider calls excluding retries.

The execution budget is:

| Stage | Added/required cells | Successful calls |
|---|---:|---:|
| Canary: complete first topic-by-trial block | 10 | 1,080 |
| Complete trial 1 across ten topics | 100 | 10,800 |
| Unconditional extension through trial 3 | 300 cumulative | 32,400 cumulative |
| Unconditional extension through trial 5 | 500 cumulative | 54,000 cumulative |

## Evidence-scoring arm

The original runner serialized `strength_estimation_method: classifier`
unconditionally, so a clean scoring arm was not expressible through the
comparison matrix alone. Before outcomes, the amendment added one explicit
`uniform_fixed` extractor mode and changed the runner to serialize the selected
memory setting. For a given newly extracted dynamic record, this mode leaves
its claim and proposition-relative polarity untouched and sets its active
`support_strength` to exactly `0.8475946340113878`. Frozen seed-record
strengths are unchanged.

The constant is the mean of 1,000 cached classifier predictions from the
pre-existing, outcome-independent AQR judgement-layer evaluation (500
affirmative and 500 negative arguments, 15 topics, fixed seed 42):

- vendored byte-identical path:
  `rebuttal/component_necessity_factorial_v1/scale_anchor/aqr_balanced_predictions.csv`;
- predictions SHA-256:
  `28fd6fafceb7adbed380602294dfe49a4353f7dcf203964b3cf1505ccc51270c`;
- source `test.csv` SHA-256:
  `18c0f0491e0206558f24dbe79b91901bd662e5ec2b1bd9d13ab740660b0c196f`;
- evaluator script SHA-256:
  `62923990c47bc8ecfaa44f5318e0e59b6a25e36c9c4c44e565adc53b0c22e04d`;
- metrics SHA-256:
  `b91d3dafec34fe3dff11a85e6e5b13a3c4876b96f53f138cff46ddfbe0f6e247`;
- ordered sample text-hash digest:
  `1101282dc122714a0351dafe18f2e9bbe6cecf822812ddf77780d7b5a7add074`.

The cached sample WA mean is `0.786509556396`; adding its registered WA bias
`0.061085077615387826` reconstructs the classifier-output mean exactly.

The registered contrast is:

`gap(full, uniform-fixed) - gap(full, graded-classifier)`.

A positive value means graded evidence scoring reduced externally measured
final disagreement relative to scale-matched unweighted evidence. This is a
whole-pathway scoring-off intervention: support strength can affect
replacement, retrieval ordering, and updating. The constant matches the
classifier's mean on the frozen AQR benchmark, not necessarily its prospective
message mean, and uniform ties can alter replacement/retrieval. No scoring
interaction will be claimed.

## Multiplicity, power, and claim boundary

The primary family now contains five contrasts. The two-sided Bonferroni alpha
is 0.01 per contrast, and the registered familywise percentile interval is 99%
with quantiles `[0.005, 0.995]`. The amended 10×5 design is the smallest audited
candidate with exact topic-sign resolution below that threshold and at least
80% conservative power for a 0.50 effect under both completed
component-relevant variance estimates.

Its conservative 80% topic-t MDE is 0.360–0.468 under those two inputs and
0.661 under the deliberately conservative empirical componentwise envelope.
A null result is not equivalence.

Persistence is only the temporal evidence-horizon factor. It is not relabeled
as the update-rule ablation: the separately locked `u=0` uptake campaign
supplies the no-update contrast.

## Realized deduplication exposure

The D1/D0 full-setting contrast now carries a mandatory outcome-independent
manipulation table. For every one of the 50 paired topic-by-trial cells, it
reports:

- D1 and D0 dynamic `added`, `ignored`, and `replaced` counts;
- final active and archived dynamic-record counts;
- the paired difference in records returned to response generation; and
- whether the sequence of SHA-256 fingerprints of the exact rendered
  retrieval contexts differs.

No raw claim or rendered context is added to telemetry. The primary analyzer
and the standalone non-importing verifier independently reconstruct the paired
table and aggregate summary from all 500 raw results. Synthetic tests cover a
known-exposure fixture and an adversarial zero-exposure fixture. Zero exposure
is a valid reportable result and never blocks unconditional continuation.

This check is essential for interpretation: if D1 rarely ignores or replaces
dynamic evidence, a null stance contrast is evidence about a weakly exposed
natural-dialogue intervention, not evidence that deduplication is unnecessary.
The telemetry does not establish duplicate-detection accuracy.

## Counterbalanced complete-block execution

The frozen execution plan contains 50 topic-by-trial blocks. Blocks are
ordered by trial and then the already-registered topic order, so the first ten
blocks are exactly the trial-1 pilot. For registered topic index \(t\),
zero-based trial index \(r\), and submission position \(j\), the cyclic
rotation offset is \((t+r)\bmod 10\), and position \(j\) receives registered
condition \((j+t+r)\bmod 10\). Every trial contains each offset exactly once,
so across all 50 blocks every condition appears exactly five times in every
one of the ten submission positions. Within each topic, the five trial
offsets are distinct, so no condition remains fixed to one submission
position across that topic's replications.

Only the current block's ten cells can be submitted. The scheduler waits for
all ten valid `results.json` completion markers before it submits any cell in
the next block. A completed cell is never resubmitted on resume: its complete
raw directory must match the previously recorded per-file SHA-256 ledger.
Marker-free partial directories remain eligible for whole-cell retry and are
quarantined by the locked runner before retry.

The immutable plan, outcome-free per-cell `execution_schedule.json` sidecar,
and runtime provenance record:

- the configured generation, shared-extraction, and evaluation backend/model
  aliases;
- SHA-256 identities of the specification, registry, lock, launcher,
  scheduler, runner, verifier, prompt, and scientific implementation sources;
- the resolved-config SHA-256 for each submitted cell;
- invocation, block, submission-window, worker, and completion timestamps; and
- the explicit fact `system_fingerprint_recorded: false`.

The scientific `results.json` is hashed for byte identity but is not parsed by
the scheduler or independent schedule verifier. The last item is a
qualification, not a substitute identifier: the configured
backend wrapper returns response content but does not expose the provider's
`system_fingerprint`. An independent verifier that imports neither scheduler
nor runner reconstructs the plan and audits aliases, source/config/raw hashes,
timestamp ordering, full-block barriers, and the false fingerprint claim.
Synthetic adversarial tests alter balanced positions, source hashes,
provenance positions, timing windows, fingerprint claims, and completed raw
bytes and require fail-closed rejection.

After a real process-level interruption, the resume launcher first runs the
frozen functional checkpoint verifier. The scheduler's outcome-blind
reconciliation then requires an exact match to that checkpoint's whole-cell
SHA-256 ledgers, validates each completed cell's separate execution sidecar,
closes stale invocation and attempt records as `interrupted`, and reconstructs
missing parent provenance only for sidecar-attested completed cells.
Marker-free cells remain eligible only for quarantine and whole-debate rerun.
Synthetic process tests send `SIGTERM` immediately after a worker writes the
atomic completion marker in both the canary and a post-canary stage; the
completed cell is recovered without resubmission or byte change. Raw-byte and
coherently re-ledgered sidecar-position tampering both fail closed.

Original complete-block implementation freeze:
`688ff0e1e67384e21ed445f953e649f05f63d18b`; predecessor:
`26f7e41019214a90b7b75bd8dad95b627cf63c07`. The pre-outcome
topic-by-trial rotation hardening is commit
`48c365a94bc65eacb3e434e99bc9c6fd499f4cfa`, and the pre-outcome
interruption-reconciliation implementation is commit
`5d780e2e823169c676dbcc32d0a7dea54f75f568`.

## Prelaunch verification requirements

Before any paid call:

1. deterministically regenerate and byte-check the 500-row registry;
2. verify the vendored calibration predictions, frozen sample identity,
   in-repo WA-plus-bias reconstruction, and exact fixed score;
3. generate all 500 configurations with `--dry-run`;
4. pass the fail-closed implementation/config verifier;
5. pass an independent runner-free reconstruction of every registry row,
   seed, factor, scoring mode, and call count;
6. run the independent post-outcome verifier only in synthetic self-test mode;
   its live mode must refuse to read outcomes until a passing
   500-result `final_verification.json` exists.
7. verify one retrieval event fingerprint per reply and exact dynamic
   active/archive accounting; independently reproduce the known-exposure and
   zero-exposure manipulation summaries.
8. generate the immutable 50-block plan twice and require byte-identical plan
   hashes, exact five-per-position balance, and a complete ten-cell canary;
9. pass the non-importing schedule verifier on the 500-config preflight and
   pass synthetic block-barrier, resume-preservation, and tamper tests.
10. pass process-level `SIGTERM` fixtures at canary and later stages, including
    marker-before-parent-ledger recovery, no completed-cell resubmission,
    byte preservation, and coherent sidecar-tamper rejection.

The independent post-outcome verifier reconstructs all five primary contrasts,
all seven registered metrics, 100,000-draw hierarchical intervals, 99%
familywise intervals, every topic estimate/sign, and all registered
leave-one-topic-out estimates without importing the runner or primary analyzer.
It also reconstructs all 50 D1/D0 manipulation rows and the aggregate
realized-exposure summary.

Exact amended artifact hashes and preflight evidence are recorded in the
implementation lock and amended dry-run verification reports.
