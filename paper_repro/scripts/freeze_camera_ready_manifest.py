"""Write checksums for the explicit Git-tracked release candidate.

Stage intended files first. This never adds files, downloads data, or rewrites
the frozen manuscript artifact manifest.
"""
from pathlib import Path
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "CAMERA_READY_MANIFEST.sha256"


def main():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    entries = []
    for name in sorted(set(tracked)):
        if not name or name == MANIFEST:
            continue
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Release entry must be a regular file: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {name}")
    (ROOT / MANIFEST).write_text("\n".join(entries) + "\n")
    print(f"Wrote {len(entries)} checksums to {MANIFEST}")


if __name__ == "__main__":
    main()
