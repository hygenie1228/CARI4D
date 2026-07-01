#!/usr/bin/env python3
"""Run scripts/run_open4dhoi.sh once per subdirectory of experiments/open4dhoi."""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def write_error(error_file: Path, summary: str) -> None:
    lines = [f"time: {datetime.now(timezone.utc).isoformat()}", summary]
    with error_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    open4dhoi_root = root / "experiments" / "open4dhoi"
    script = root / "scripts" / "run_open4dhoi.sh"

    if not open4dhoi_root.is_dir():
        print(f"Missing directory: {open4dhoi_root}", file=sys.stderr)
        sys.exit(1)
    if not script.is_file():
        print(f"Missing script: {script}", file=sys.stderr)
        sys.exit(1)

    exp_dirs = sorted(
        p for p in open4dhoi_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not exp_dirs:
        print(f"No experiment folders under {open4dhoi_root}", file=sys.stderr)
        sys.exit(1)

    failures: list[str] = []
    error_file = root / "error.txt"

    exp_dirs = exp_dirs 
    for exp in exp_dirs:
        # 1 cup-20250825_201936
        # 2 block-20251018_212729
        # 3 chair-20250909_133115
        # 4 brush-20250901_204341
        # 5 dumbell-20250729_235623

        if "dumbell-20250729_235623" not in str(exp):
            continue

        rel = exp.relative_to(root)
        human_npz = exp / "human" / "human_params.npz"
        object_npz = exp / "object" / "object_params.npz"
        if human_npz.is_file() and object_npz.is_file():
            print(f"==> {rel} (skip: params already exist)", flush=True)
            continue
        print(f"==> {rel}", flush=True)

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            r = subprocess.run(
                ["bash", str(script), str(rel)],
                cwd=root,
                env=env,
            )
            if r.returncode != 0:
                print(
                    f"!!! {rel} failed with exit code {r.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                write_error(
                    error_file,
                    f"sample: {rel}\nexit code: {r.returncode}",
                )
                failures.append(str(rel))
                continue
        except Exception as e:
            print(f"!!! {rel} failed to start: {e}", file=sys.stderr, flush=True)
            write_error(error_file, f"sample: {rel}\nfailed to start: {e}")
            failures.append(str(rel))
            continue

    if failures:
        print("\nFailed samples:", file=sys.stderr)
        for f in failures:
            print(f"- {f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
