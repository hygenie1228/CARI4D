#!/usr/bin/env python3
"""Recenter all experiments/behave/*/object/model.obj so AABB center is (0,0,0)."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "experiments" / "behave"
# Skip recenter when AABB center is already near the origin (per-axis).
CENTER_SKIP_EPS = 1e-4


def bbox_center(path: Path) -> tuple[float, float, float] | None:
    xmin = ymin = zmin = math.inf
    xmax = ymax = zmax = -math.inf
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            xmin, xmax = min(xmin, x), max(xmax, x)
            ymin, ymax = min(ymin, y), max(ymax, y)
            zmin, zmax = min(zmin, z), max(zmax, z)
            n += 1
    if n == 0:
        return None
    return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))


def recenter_file(path: Path, cx: float, cy: float, cz: float) -> None:
    d = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=d,
        delete=False,
        prefix=".model_recenter_",
        suffix=".tmp",
    ) as out:
        tmp = Path(out.name)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.startswith("v "):
                        out.write(line)
                        continue
                    parts = line.split()
                    if len(parts) != 7:
                        out.write(line)
                        continue
                    x = float(parts[1]) - cx
                    y = float(parts[2]) - cy
                    z = float(parts[3]) - cz
                    out.write(
                        f"v {x:.6f} {y:.6f} {z:.6f} {parts[4]} {parts[5]} {parts[6]}\n"
                    )
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


def main() -> int:
    if not ROOT.is_dir():
        print(f"Missing directory: {ROOT}", file=sys.stderr)
        return 1
    objs = sorted(ROOT.glob("*/object/model.obj"))
    if not objs:
        print(f"No model.obj under {ROOT}/*/object/", file=sys.stderr)
        return 1
    for obj in objs:
        c = bbox_center(obj)
        if c is None:
            print(f"skip (no v-lines): {obj}", file=sys.stderr)
            continue
        cx, cy, cz = c
        if max(abs(cx), abs(cy), abs(cz)) < CENTER_SKIP_EPS:
            print(
                f"{obj.relative_to(ROOT.parent.parent)}  skip (|center|_inf < {CENTER_SKIP_EPS:g})"
            )
            continue
        recenter_file(obj, cx, cy, cz)
        print(f"{obj.relative_to(ROOT.parent.parent)}  center was ({cx:.6f}, {cy:.6f}, {cz:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
