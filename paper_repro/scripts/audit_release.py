#!/usr/bin/env python3
"""Audit the anonymous reviewer artifact for common release residue."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()

TEXT_SUFFIXES = {
    ".cfg",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".tex",
}

FORBIDDEN_FILE_NAMES = {
    ".DS_Store",
    ".env",
}

FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".mplconfig",
    "raw_runs",
}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
}

FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
}

PATH_PATTERNS = [
    re.compile(r"/Users/[^\\s:'\")]+"),
    re.compile(r"/home/[^\\s:'\")]+"),
    re.compile(r"/private/var/[^\\s:'\")]+"),
]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{24,}"),
]

ALLOWED_SECRET_PLACEHOLDERS = {
    "sk-dummy",
    "sk-example",
    "sk-your-api-key-here",
    "sk-or-your-key-here",
}


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        rel = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def check_filesystem_residue(files: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        if path.name in FORBIDDEN_FILE_NAMES:
            issues.append(f"forbidden file: {rel}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            issues.append(f"forbidden generated file: {rel}")
        if any(part in FORBIDDEN_DIR_NAMES for part in rel.parts):
            issues.append(f"forbidden generated/raw directory content: {rel}")
    return issues


def check_text_residue(files: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in files:
        if not is_text_file(path):
            continue
        if path.resolve() == THIS_FILE:
            continue
        rel = path.relative_to(REPO_ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for pattern in PATH_PATTERNS:
            for match in pattern.findall(text):
                issues.append(f"absolute local path in {rel}: {match}")

        for pattern in SECRET_PATTERNS:
            for match in pattern.findall(text):
                if match not in ALLOWED_SECRET_PLACEHOLDERS:
                    issues.append(f"possible secret in {rel}: {match[:7]}...")
    return issues


def main() -> int:
    files = iter_files()
    issues = check_filesystem_residue(files)
    issues.extend(check_text_residue(files))

    if issues:
        print("Release audit failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print(f"Release audit passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
