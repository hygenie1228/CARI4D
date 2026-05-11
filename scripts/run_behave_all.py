#!/usr/bin/env python3
"""Run scripts/run_behave.sh once per subdirectory of experiments/behave."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    behave = root / "experiments" / "behave"
    script = root / "scripts" / "run_behave.sh"

    if not behave.is_dir():
        print(f"Missing directory: {behave}", file=sys.stderr)
        sys.exit(1)
    if not script.is_file():
        print(f"Missing script: {script}", file=sys.stderr)
        sys.exit(1)

    exp_dirs = sorted(
        p for p in behave.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not exp_dirs:
        print(f"No experiment folders under {behave}", file=sys.stderr)
        sys.exit(1)

    exp_dirs = exp_dirs[::3]
    failures: list[str] = []
    for exp in exp_dirs:
        rel = exp.relative_to(root)
        human_npz = exp / "human" / "human_params.npz"
        object_npz = exp / "object" / "object_params.npz"
        if human_npz.is_file() and object_npz.is_file():
            print(f"==> {rel} (skip: params already exist)", flush=True)
            continue
        print(f"==> {rel}", flush=True)

        try:
            r = subprocess.run(["bash", str(script), str(rel)], cwd=root)
            if r.returncode != 0:
                print(
                    f"!!! {rel} failed with exit code {r.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                failures.append(str(rel))
                continue
        except Exception as e:
            print(f"!!! {rel} failed to start: {e}", file=sys.stderr, flush=True)
            failures.append(str(rel))
            continue

    if failures:
        print("\nFailed samples:", file=sys.stderr)
        for f in failures:
            print(f"- {f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
