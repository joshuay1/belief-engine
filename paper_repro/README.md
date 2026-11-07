# Reproduction tooling

For installation and commands, start with the [reproduction guide](../REPRODUCING.md).

- `scripts/`: verification and test entry points used by `make check`.
- `manifest.sha256`: checksums for the current release candidate.
- `requirements-verified-py311.txt`: dependency versions used for verification.
- `provenance/`: source hashes and original reviewer-snapshot records.
- [CAMERA_READY.md](CAMERA_READY.md): maintainer notes and remaining release work.
- [CODE_VALIDATION.json](CODE_VALIDATION.json): recorded verification results.

Run the public commands from the repository root. The original May snapshot and
its superseded documentation remain available in the first Git commit.
