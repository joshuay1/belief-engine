# Paper Reproducibility Notes

The top-level `README.md` is the reviewer entry point. This directory contains
the files that define and verify the paper-facing reproducibility boundary.

## Reviewer Commands

From the repository root:

```bash
make -f paper_repro/Makefile reviewer-check
```

Useful narrower checks:

```bash
make -f paper_repro/Makefile smoke
make -f paper_repro/Makefile audit-release
make -f paper_repro/Makefile check-artifacts
make -f paper_repro/Makefile check-release-manifest
```

## Files

- `Makefile`: reviewer and maintainer helper targets.
- `manifest.yaml`: paper object to artifact/source mapping.
- `PAPER_EXPERIMENT_INVENTORY.md`: human-readable experiment inventory.
- `RELEASE_ALLOWLIST.md`: release boundary.
- `RELEASE_AUDIT.md`: current release audit notes.
- `scripts/check_artifacts.py`: checksum verifier.
- `scripts/reviewer_check.py`: Python version, syntax, and import checks.
- `scripts/audit_release.py`: anonymity/release-residue audit.

The byte-level artifact verification target is
`validation/paper_artifacts/MANIFEST.sha256`.
