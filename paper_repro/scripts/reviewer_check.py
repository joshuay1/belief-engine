#!/usr/bin/env python3
"""Run reviewer-safe cold checks without creating bytecode caches."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}

IMPORT_CHECKS = [
    "src.cli",
    "src.simulation.belief_engine",
    "validation.fitting.metrics",
    "validation.evaluation.common",
    "validation.scripts.prepare_paper_artifacts",
    "validation.scripts.build_6panel_topic_grids_and_metrics",
    "validation.scripts.plot_model_sensitivity_comparison",
    "validation.scripts.plot_top_matched_movers",
    "validation.scripts.summarize_debate_rmse_uncertainty",
    "paper_repro.scripts.check_artifacts",
]


def iter_python_files() -> list[Path]:
    paths: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        paths.append(path)
    return sorted(paths)


def check_python_version() -> None:
    if sys.version_info < (3, 11):
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise RuntimeError(f"Python >=3.11 is required; found {version}")
    version = ".".join(str(part) for part in sys.version_info[:3])
    print(f"Python: {version}")


def check_syntax() -> None:
    failures: list[tuple[Path, SyntaxError]] = []
    files = iter_python_files()
    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            failures.append((path, exc))

    if failures:
        print("\nSyntax errors:")
        for path, exc in failures:
            rel = path.relative_to(REPO_ROOT)
            print(f"  {rel}:{exc.lineno}: {exc.msg}")
        raise RuntimeError("syntax check failed")

    print(f"Syntax: {len(files)} Python files OK")


def check_imports() -> None:
    failures: list[tuple[str, Exception]] = []
    for module in IMPORT_CHECKS:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append((module, exc))

    if failures:
        print("\nImport failures:")
        for module, exc in failures:
            print(f"  {module}: {type(exc).__name__}: {exc}")
        raise RuntimeError("import check failed")

    print(f"Imports: {len(IMPORT_CHECKS)} key modules OK")


def main() -> int:
    check_python_version()
    check_syntax()
    check_imports()
    print("Reviewer code check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
