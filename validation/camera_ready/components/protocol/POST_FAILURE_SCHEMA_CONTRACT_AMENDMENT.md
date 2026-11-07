# Post-failure structured-output contract amendment

## Status and timing

This amendment was frozen after the first provider-backed implementation
canary stopped, and before any component-factorial effect estimate was computed
or used. The failed canary produced nine valid `results.json` markers and one
marker-free failed cell, so it contained zero complete ten-condition blocks
under the registered complete-case rule.

The repair decision used only operational information: the completion-marker
count, the failed run identity, and the exact structured-output validation
error. No stance, trajectory, condition contrast, direction, or significance
value from the nine completed cells was used to select or define the repair.

## Exact failure

- Predecessor source commit:
  `152774a2bfa37245d59ea25cadd12513dacc523d`
- Failed run:
  `social_media__be_s1_r1_p1_d1_uniform__trial_01`
- Failed condition:
  `be_s1_r1_p1_d1_uniform`
- Exact error:
  `Structured extraction stage 'distillation' failed after 2 attempt(s):
  arguments must contain at most 12 items, got 13`
- Completed cells in the aborted canary: 9 of 10
- Complete analyzable blocks in the aborted canary: 0

The prompt already instructed the provider to return at most twelve arguments,
and the local validator already rejected arrays longer than twelve. The strict
JSON schema supplied to the provider omitted the corresponding
`maxItems: 12` constraint. The provider therefore returned a schema-valid
thirteen-item array that failed the local contract on both configured attempts.

## Repair

The repaired implementation adds only `maxItems: 12` to the `arguments` array
inside `argument_distillation_v1`. It changes no prompt text, local validator,
claim definition, factor, condition, topic, seed, endpoint, contrast,
multiplicity rule, continuation rule, or call budget.

- Repair implementation commit:
  `247d408de0a320e0badcba6e7adaa1d3d7106d9c`
- Repaired extractor SHA-256:
  `c2fa51bbe865a270aeb3e2b3be9755454e1d9cdb3b9233695d8d467bcf4551e7`
- Contract-test SHA-256:
  `83c556cef09b76827159029eddd2f5020162a7c9b5a5e1098bfa64a2996a6e28`

## Restart and exclusion rule

The nine completed predecessor cells and the failed partial tree are preserved
as an aborted implementation canary. They are excluded from every scientific
analysis and will not be copied into or pooled with the repaired campaign.

A byte-identical local archive of the stopped predecessor is held at
`../agent_mem_artifact_packaging/rebuttal/results/execution/component_factorial_failed_canary_20260726_before_schema_repair`.
It contains 587 files totaling 8,244,226 file bytes. The SHA-256 of the
byte-sorted relative-path file-hash ledger is
`5827a96ca367f0ba025cc77bc7c6c754d2bfa15c3bf0dd9f5439541491e22fe1`.
The archived execution plan, execution provenance, and manifest have SHA-256
values `34b2bd4a136f58427fc8acd2d3471ebe9ae101071815548f7dd4a970e51e2617`,
`f187b3895dcddd2c217905da44c0a49682c3e048afd0d771551d3e29290e3690`,
and `1b95aec877231f00ac7560e024ee4ca6c56bc791bfdb7385cfaa95ccb34485a9`,
respectively. These values match the original stopped root.

An initial repaired launch root was later excluded after the separate
spawned-worker import-path failure documented in
`POST_FAILURE_WORKER_IMPORT_PATH_AMENDMENT.md`. The permanent repaired campaign
starts in
`rebuttal/results/component_necessity_factorial_v1_schema_contract_repair_restart_v3`,
regenerates all 500 registered configurations, and reruns the complete first
ten-condition block. Its default analysis root is
`rebuttal/results/analyses/component_necessity_factorial_v1_schema_contract_repair_restart_v3`.
If that canary passes, the already registered pilot, mid, and full stages
continue unconditionally. Every scientific contrast still requires all ten
conditions in each topic-by-trial block. The repaired campaign retains the same
registered seeds and scientific settings; only the provider-enforced schema now
matches the pre-existing prompt and validator.

This is a post-failure, outcome-blind implementation-contract repair, not an
untouched first execution. The final report must disclose the aborted canary,
the schema mismatch, the full restart, and the exclusion of all predecessor
cells.
