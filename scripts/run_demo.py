#!/usr/bin/env python3
"""Run CoCoNet demo for a BEHAVE experiment directory.

Equivalent to run_behave.sh Step 5 (lines 41-49), but outputs under:
  <exp_dir>/cari4d/**.pth
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import subprocess
import sys
from typing import List, Tuple
import tempfile

import h5py
import numpy as np
from tqdm import tqdm


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

    try:
        with open(obj_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except PermissionError:
        print(f"[obj-preprocess] skipped: no write permission for {obj_path}")
        return

    print(
        f"[obj-preprocess] recentered bbox to origin and overwrote OBJ "
        f"(old_center=({cx:.6e}, {cy:.6e}, {cz:.6e}), eps={eps:.1e})"
    )


def build_masks_h5_from_processed(
    video_prefix: str,
    cam_id: int,
    human_mask_mp4: str,
    object_mask_mp4: str,
    masks_h5_path: str,
) -> None:
    """Build CARI4D-compatible h5 mask file from processed videos."""

    def _video_hw(path: str) -> tuple[int, int]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            path,
        ]
        out = subprocess.check_output(cmd, text=True)
        streams = json.loads(out).get("streams", [])
        if not streams:
            raise RuntimeError(f"ffprobe found no video stream: {path}")
        w = int(streams[0]["width"])
        h = int(streams[0]["height"])
        if w <= 0 or h <= 0:
            raise RuntimeError(f"invalid video shape from ffprobe: {path}")
        return h, w

    def _spawn_gray_reader(path: str) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    h_h, w_h = _video_hw(human_mask_mp4)
    h_o, w_o = _video_hw(object_mask_mp4)
    if (h_h, w_h) != (h_o, w_o):
        raise RuntimeError(f"mask shape mismatch: human={(h_h, w_h)} object={(h_o, w_o)}")

    frame_bytes = h_h * w_h
    proc_h = _spawn_gray_reader(human_mask_mp4)
    proc_o = _spawn_gray_reader(object_mask_mp4)
    if proc_h.stdout is None or proc_o.stdout is None:
        raise RuntimeError("ffmpeg stdout is unavailable")

    os.makedirs(osp.dirname(masks_h5_path), exist_ok=True)
    if osp.isfile(masks_h5_path):
        os.remove(masks_h5_path)

    frame_idx = 0
    try:
        with h5py.File(masks_h5_path, "w") as h5:
            pbar = tqdm(desc="processed masks -> h5", unit="frame")
            while True:
                buf_h = proc_h.stdout.read(frame_bytes)
                buf_o = proc_o.stdout.read(frame_bytes)
                if not buf_h or not buf_o:
                    break
                if len(buf_h) != frame_bytes or len(buf_o) != frame_bytes:
                    raise RuntimeError(
                        f"incomplete frame read at {frame_idx}: human={len(buf_h)}, object={len(buf_o)}"
                    )
                gh = np.frombuffer(buf_h, dtype=np.uint8).reshape(h_h, w_h) > 127
                go = np.frombuffer(buf_o, dtype=np.uint8).reshape(h_h, w_h) > 127
                h5.create_dataset(
                    f"{video_prefix}/{frame_idx:06d}-k{cam_id}.person_mask.png",
                    data=gh,
                    compression="gzip",
                    compression_opts=3,
                )
                h5.create_dataset(
                    f"{video_prefix}/{frame_idx:06d}-k{cam_id}.obj_rend_mask.png",
                    data=go,
                    compression="gzip",
                    compression_opts=3,
                )
                frame_idx += 1
                pbar.update(1)
            pbar.close()
    finally:
        proc_h.stdout.close()
        proc_o.stdout.close()
        proc_h.wait(timeout=30)
        proc_o.wait(timeout=30)
    if frame_idx <= 0:
        raise RuntimeError("no frames were read from processed mask videos")


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
    human_mask_mp4 = osp.join(exp_dir, "processed", "human_mask.mp4")
    object_mask_mp4 = osp.join(exp_dir, "processed", "object_mask.mp4")
    depth_mp4 = osp.join(exp_dir, "processed", "depth.mp4")
    nlf_ud_dir = osp.join(cari4d, "nlf-2unidepth")
    fp_dir = osp.join(cari4d, "fp-hy3d3-unidepth")
    hy3d_mesh = osp.join(exp_dir, "object", "model.obj")
    video_src = osp.join(exp_dir, "video.mp4")
    tmp_root = osp.join(tempfile.gettempdir(), "cari4d_run_demo", osp.basename(exp_dir))
    videos_dir = osp.join(tmp_root, "videos")
    masks_dir = osp.join(tmp_root, "masks")
    video = osp.join(videos_dir, f"{video_prefix}.{cam_id}.color.mp4")
    depth_reg = osp.join(videos_dir, f"{video_prefix}.{cam_id}.depth-reg.mp4")
    masks_h5 = osp.join(masks_dir, f"{video_prefix}_masks_k{cam_id}.h5")

    required_paths = [hy3d_mesh, human_mask_mp4, object_mask_mp4, depth_mp4, fp_dir, nlf_ud_dir, video_src]
    missing = [p for p in required_paths if not osp.exists(p)]
    if missing:
        print("missing required paths:", file=sys.stderr)
        for path in missing:
            print(f" - {path}", file=sys.stderr)
        sys.exit(1)

    center_obj_bbox_inplace(hy3d_mesh, eps=1e-4)
    build_masks_h5_from_processed(
        video_prefix=video_prefix,
        cam_id=cam_id,
        human_mask_mp4=human_mask_mp4,
        object_mask_mp4=object_mask_mp4,
        masks_h5_path=masks_h5,
    )
    os.makedirs(videos_dir, exist_ok=True)
    for src, dst in ((video_src, video), (depth_mp4, depth_reg)):
        if osp.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

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
