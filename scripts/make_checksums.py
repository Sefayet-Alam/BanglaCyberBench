#!/usr/bin/env python3
"""Create a deterministic SHA-256 manifest for a release snapshot."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--output", default="SHA256SUMS")
    args = p.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    ignored = {".git", ".ipynb_checkpoints", "SHA256SUMS"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        if path.resolve() == output:
            continue
        files.append(path)
    with output.open("w", encoding="utf-8") as f:
        for path in files:
            f.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")
    print(f"Wrote {len(files)} checksums to {output}")


if __name__ == "__main__":
    main()
