#!/usr/bin/env python3
"""Run scripts/run_intercap.sh once per subdirectory of experiments/intercap

InterCap-style folders do not encode a Kinect/view id in the directory name. Use the
environment variable ``ICAP_VIEW`` (default ``0`` in ``run_intercap.sh``) for the
camera index passed to staging, FP, CoCoNet, opt, and NPZ export.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    intercap_root = root / "experiments" / "intercap"
    script = root / "scripts" / "run_intercap.sh"

    if not intercap_root.is_dir():
        print(f"Missing directory: {intercap_root}", file=sys.stderr)
        sys.exit(1)
    if not script.is_file():
        print(f"Missing script: {script}", file=sys.stderr)
        sys.exit(1)

    exp_dirs = sorted(
        p for p in intercap_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not exp_dirs:
        print(f"No experiment folders under {intercap_root}", file=sys.stderr)
        sys.exit(1)

    exp_dirs = exp_dirs[::-1]
    failures: list[str] = []
    for exp in exp_dirs:
        rel = exp.relative_to(root)
        human_npz = exp / "human" / "human_params.npz"
        object_npz = exp / "object" / "object_params.npz"
        if human_npz.is_file() and object_npz.is_file():
            print(f"==> {rel} (skip: params already exist)", flush=True)
            continue
        print(f"==> {rel}", flush=True)

        env = os.environ.copy()
        try:
            r = subprocess.run(["bash", str(script), str(rel)], cwd=root, env=env)
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
