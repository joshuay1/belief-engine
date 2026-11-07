#!/usr/bin/env python
"""Verify the selected paper artifact checksum manifest.

This intentionally checks only the compact artifact bundle, not the full
exploratory validation tree.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "paper_artifacts" / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            expected, rel_path = stripped.split(None, 1)
        except ValueError as exc:
            raise ValueError(f"Malformed line {line_number} in {path}") from exc
        entries.append((expected, REPO_ROOT / rel_path.strip()))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="SHA256 manifest to verify.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list manifest entries; do not compute checksums.",
    )
    args = parser.parse_args()

    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    entries = read_manifest(manifest)
    print(f"Manifest: {manifest.relative_to(REPO_ROOT)}")
    print(f"Entries: {len(entries)}")

    if args.list_only:
        for _, path in entries:
            print(path.relative_to(REPO_ROOT))
        return 0

    missing: list[Path] = []
    mismatched: list[tuple[Path, str, str]] = []

    for expected, path in entries:
        if not path.exists():
            missing.append(path)
            continue
        actual = sha256(path)
        if actual != expected:
            mismatched.append((path, expected, actual))

    if missing:
        print("\nMissing files:")
        for path in missing:
            print(f"  {path.relative_to(REPO_ROOT)}")

    if mismatched:
        print("\nChecksum mismatches:")
        for path, expected, actual in mismatched:
            print(f"  {path.relative_to(REPO_ROOT)}")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")

    if missing or mismatched:
        print("\nArtifact check failed.")
        return 1

    print("Artifact check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

