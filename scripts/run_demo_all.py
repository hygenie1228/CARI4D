#!/usr/bin/env python3
"""Run scripts/run_demo_test.sh once per subdirectory of experiments/demo_test."""

import os
import subprocess
import sys
from pathlib import Path

# One-off: seed human pose/shape from exp2 for skateboard-demo_3 instead of NLF.
_DEMO_HUMAN_INIT_OVERRIDE = {
    "skateboard-demo_3": "exp2/demo_test/skateboard-demo_3/human/human_params_init.npz",
}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    demo_root = root / "experiments" / "demo_test"
    script = root / "scripts" / "run_demo_test.sh"

    if not demo_root.is_dir():
        print(f"Missing directory: {demo_root}", file=sys.stderr)
        sys.exit(1)
    if not script.is_file():
        print(f"Missing script: {script}", file=sys.stderr)
        sys.exit(1)

    exp_dirs = sorted(
        p for p in demo_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not exp_dirs:
        print(f"No experiment folders under {demo_root}", file=sys.stderr)
        sys.exit(1)

    failures: list[str] = []
    exp_dirs = exp_dirs
    for exp in exp_dirs:
        if "skateboard-demo_3" not in str(exp):
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
            init_rel = _DEMO_HUMAN_INIT_OVERRIDE.get(exp.name)
            if init_rel is not None:
                init_path = root / init_rel
                if not init_path.is_file():
                    print(f"!!! missing human init override: {init_path}", file=sys.stderr)
                    failures.append(str(rel))
                    continue
                env["DEMO_HUMAN_INIT_NPZ"] = str(init_path)
                print(f"    using human init override: {init_rel}", flush=True)
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
