#!/usr/bin/env python3
"""Run scripts/run_open4dhoi.sh once per subdirectory of experiments/open4dhoi."""

import subprocess
import sys
from pathlib import Path

# skateboard_0359_026
# skateboard_0339_010
# skateboard_0045_002

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
    
    exp_dirs = exp_dirs[1::2]
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
