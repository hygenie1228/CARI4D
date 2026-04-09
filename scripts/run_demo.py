#!/usr/bin/env python3
"""Run CoCoNet demo for a BEHAVE experiment directory.

Equivalent to run_behave.sh Step 5 (lines 41-49), but outputs under:
  <exp_dir>/cari4d/**.pth
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import subprocess
import sys
from typing import List, Tuple


def parse_exp_dir(exp_dir: str) -> tuple[str, int]:
    base = osp.basename(exp_dir.rstrip("/"))
    m = re.match(r"^(.+)_(\d+)$", base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


def _parse_obj_vertex(line: str) -> Tuple[float, float, float] | None:
    if not line.startswith("v "):
        return None
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    try:
        return float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None


def center_obj_bbox_inplace(obj_path: str, eps: float = 1e-4) -> None:
    with open(obj_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    vertices: List[Tuple[int, float, float, float]] = []
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    for idx, line in enumerate(lines):
        parsed = _parse_obj_vertex(line)
        if parsed is None:
            continue
        x, y, z = parsed
        vertices.append((idx, x, y, z))
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    if not vertices:
        raise ValueError(f"No valid vertex lines found in OBJ: {obj_path}")

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    cz = 0.5 * (min_z + max_z)
    max_abs_center = max(abs(cx), abs(cy), abs(cz))

    if max_abs_center < eps:
        print(
            f"[obj-preprocess] skipped: bbox center already near origin "
            f"(center=({cx:.6e}, {cy:.6e}, {cz:.6e}), eps={eps:.1e})"
        )
        return

    for idx, x, y, z in vertices:
        nx = x - cx
        ny = y - cy
        nz = z - cz
        lines[idx] = f"v {nx:.10f} {ny:.10f} {nz:.10f}\n"

    with open(obj_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(
        f"[obj-preprocess] recentered bbox to origin and overwrote OBJ "
        f"(old_center=({cx:.6e}, {cy:.6e}, {cz:.6e}), eps={eps:.1e})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment directory.",
    )
    args = parser.parse_args()

    exp_dir = osp.abspath(args.exp_dir)
    video_prefix, cam_id = parse_exp_dir(exp_dir)

    cari4d = osp.join(exp_dir, "cari4d")
    videos_dir = osp.join(cari4d, "videos")
    human_mask_mp4 = osp.join(exp_dir, "processed", "human_mask.mp4")
    object_mask_mp4 = osp.join(exp_dir, "processed", "object_mask.mp4")
    nlf_ud_dir = osp.join(cari4d, "nlf-2unidepth")
    fp_dir = osp.join(cari4d, "fp-hy3d3-unidepth")
    hy3d_mesh = osp.join(exp_dir, "object", "model.obj")
    video = osp.join(videos_dir, f"{video_prefix}.{cam_id}.color.mp4")

    required_paths = [hy3d_mesh, human_mask_mp4, object_mask_mp4, fp_dir, nlf_ud_dir, video]
    missing = [p for p in required_paths if not osp.exists(p)]
    if missing:
        print("missing required paths:", file=sys.stderr)
        for path in missing:
            print(f" - {path}", file=sys.stderr)
        sys.exit(1)

    center_obj_bbox_inplace(hy3d_mesh, eps=1e-4)

    os.makedirs(cari4d, exist_ok=True)

    cmd = [
        sys.executable,
        "run_horefine.py",
        "config=learning/configs/cari4d-release.yml",
        "split_file=splits/demo-behave.json",
        "use_sel_view=True",
        "render_video=True",
        "identifier=_demo",
        "use_intermediate=True",
        "data_name=test-only",
        f"hy3d_meshes_root={hy3d_mesh}",
        f"masks_root={human_mask_mp4},{object_mask_mp4}",
        f"fp_root={fp_dir}",
        f"nlf_root={nlf_ud_dir}",
        f"video={video}",
        f"cam_id={cam_id}",
        f"outpath={cari4d}",
    ]

    print("running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=osp.dirname(osp.dirname(__file__)))


if __name__ == "__main__":
    main()
