# Post-failure flat-extraction completion-audit amendment

## Status and timing

This amendment was frozen after the schema- and worker-repaired canary
completed all ten registered conditions. The independent schedule verifier
passed, but the scientific verifier rejected the four
`whole_message_json_v1` conditions because their post-run metadata omitted the
redundant `complete` field.

The repair used only extraction audit counters and verifier errors. No stance
trajectory, endpoint, condition contrast, direction, significance value, or
effect estimate was inspected or used to select the repair.

## Exact metadata mismatch

Each of the four flat conditions recorded:

- 30 top-level extraction calls;
- 30 successful classification passes;
- zero extraction failures;
- 30 single-record extraction calls;
- zero zero-record or multi-record extraction calls; and
- 30 emitted records.

The generic runner enriched extraction metadata with
`expected_extract_calls`, `completed_extract_calls`, and `complete` only when
`output_mode == "structured_json_v1"`. The same strict completion contract was
therefore absent from otherwise complete `whole_message_json_v1` metadata.
The verifier correctly remained fail-closed and returned four
`extraction incomplete` errors.

## Repair

The generic runner now applies its existing post-run extraction completion
audit to both registered record modes:

- `structured_json_v1`; and
- `whole_message_json_v1`.

A regression test requires both modes, but not the legacy text mode, to receive
the strict completion audit. This changes only post-run metadata. It changes no
provider call, prompt, generated response, extraction/classification request,
memory operation, factor, condition, topic, seed, endpoint, contrast,
multiplicity rule, continuation rule, or call budget.

- Flat completion-audit repair commit:
  `133dd721840c8522289bb7f4eeb12c6b8a6c6798`
- Repair predecessor commit:
  `0bada6d8688caea311d64133cb4f06c8dd8953fb`
- Repaired generic-runner SHA-256:
  `1fb255feb7b06b06862002de252c0e3161dad9073ab8d69f6c88b0136fb1c057`
- Regression-test SHA-256:
  `335c6689e867ee1d7c23511eee205b022838012d1d156c3f6069884fb56ac877`

## Exclusion, preservation, and final restart

The completed v2 canary is preserved but excluded rather than backfilled or
mixed with differently enriched result metadata. Its original root is
`rebuttal/results/component_necessity_factorial_v1_schema_contract_repair_restart_v2`.
A byte-identical archive is held at
`../agent_mem_artifact_packaging/rebuttal/results/execution/component_factorial_v2_flat_verifier_contract_canary_20260726`.

The archive contains 589 files totaling 8,661,500 file bytes. The SHA-256 of
its byte-sorted relative-path file-hash ledger is
`37f7799f64b4d4a7ee0af37d6ed070207292ab25d1b91352baa3975facb63bac`.
Its execution plan, execution provenance, and manifest have SHA-256 values
`c0aa94fafd4cbce4fe19613b68393ae13faf446579fd7c54747b346ab5a7f063`,
`0751f6de8ecca23cb0cdd0eca1a8f51223747a0cb72114a750557f1131b71080`,
and `3f7883f3e2c517650b99f7174b3745f31c4329e0d2a5a76140fa5838fe18e0fe`,
respectively.

The final campaign starts from
`rebuttal/results/component_necessity_factorial_v1_schema_contract_repair_restart_v3`
and reruns all 500 registered cells, beginning with the full ten-condition
canary. Its analysis root is
`rebuttal/results/analyses/component_necessity_factorial_v1_schema_contract_repair_restart_v3`.
No file from any excluded predecessor root may enter this final campaign or
analysis.
