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


def parse_exp_dir(exp_dir: str) -> tuple[str, int]:
    base = osp.basename(exp_dir.rstrip("/"))
    m = re.match(r"^(.+)_(\d+)$", base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


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
    masks_dir = osp.join(cari4d, "masks")
    nlf_ud_dir = osp.join(cari4d, "nlf-2unidepth")
    fp_dir = osp.join(cari4d, "fp-hy3d3-unidepth")
    hy3d_mesh = osp.join(exp_dir, "object", "model.obj")
    video = osp.join(videos_dir, f"{video_prefix}.{cam_id}.color.mp4")

    required_paths = [hy3d_mesh, masks_dir, fp_dir, nlf_ud_dir, video]
    missing = [p for p in required_paths if not osp.exists(p)]
    if missing:
        print("missing required paths:", file=sys.stderr)
        for path in missing:
            print(f" - {path}", file=sys.stderr)
        sys.exit(1)

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
        f"masks_root={masks_dir}",
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
