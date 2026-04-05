#!/usr/bin/env python3
"""Build exp_dir/cari4d/videos/* and masks/*.h5 from video.mp4 + processed/* (wild_video / kinect 0)."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import os.path as osp
import cv2
import h5py
import joblib
import numpy as np
from tqdm import tqdm
from videoio import Uint16Writer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate videos/masks/hy3d symlink even if outputs already exist.",
    )
    args = p.parse_args()
    exp_dir = osp.abspath(args.exp_dir)

    base = osp.basename(exp_dir.rstrip("/"))
    m = re.match(r"^(.+)_(\d+)$", base)
    video_prefix = m.group(1) if m else base
    kid = 0

    npz_path = osp.join(exp_dir, "human", "human_params.npz")
    src_color = osp.join(exp_dir, "video.mp4")
    d_human = osp.join(exp_dir, "processed", "human_mask.mp4")
    d_obj = osp.join(exp_dir, "processed", "object_mask.mp4")
    d_bgr = osp.join(exp_dir, "processed", "depth.mp4")

    work_root = osp.join(exp_dir, "cari4d")
    videos_dir = osp.join(work_root, "videos")
    color_out = osp.join(videos_dir, f"{video_prefix}.{kid}.color.mp4")
    pkl_out = color_out.replace(".mp4", ".pkl")
    depth_out = osp.join(videos_dir, f"{video_prefix}.{kid}.depth-reg.mp4")
    masks_dir = osp.join(work_root, "masks")
    h5_path = osp.join(masks_dir, f"{video_prefix}_masks_k{kid}.h5")
    src_model = osp.join(exp_dir, "object", "model.obj")
    hy_sub = osp.join(work_root, "hy3d_staged", f"{video_prefix}_export")
    dst_align = osp.join(hy_sub, f"{video_prefix}_align.obj")

    if not args.force:
        core_ok = (
            osp.isfile(color_out)
            and osp.isfile(pkl_out)
            and osp.isfile(depth_out)
            and osp.isfile(h5_path)
        )
        hy_ok = not osp.isfile(src_model) or osp.lexists(dst_align)
        if core_ok and hy_ok:
            print(f"skip (outputs exist): {work_root} — use --force to regenerate")
            return

    assert osp.isfile(src_color), src_color
    assert osp.isfile(d_human) and osp.isfile(d_obj), (d_human, d_obj)
    assert osp.isfile(d_bgr), d_bgr

    if osp.isfile(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        intr = np.asarray(z["intrinsics"], dtype=np.float64).reshape(-1)[:4]
    else:
        intr = np.array([979.784, 979.840, 1018.952, 779.486], dtype=np.float64)
        print(
            "warning: human/human_params.npz missing; using default BEHAVE intrinsics. "
            "FoundationPose expects human_params.npz in exp_dir."
        )

    os.makedirs(videos_dir, exist_ok=True)
    shutil.copy2(src_color, color_out)
    joblib.dump(
        {"fx": float(intr[0]), "fy": float(intr[1]), "cx": float(intr[2]), "cy": float(intr[3])},
        pkl_out,
    )

    cap_r = cv2.VideoCapture(d_bgr)
    fps = cap_r.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
    depth_writer = Uint16Writer(depth_out, (W, H), fps=int(round(fps)))
    for _ in tqdm(range(max(n, 1)), desc="processed/depth.mp4 -> depth-reg"):
        ret, fr = cap_r.read()
        if not ret or fr is None:
            break
        _b, g, r = cv2.split(fr)
        d16 = (r.astype(np.uint32) << 8) | g.astype(np.uint32)
        depth_writer.write(d16.astype(np.uint16))
    cap_r.release()
    depth_writer.close()
    print("wrote", depth_out)

    os.makedirs(masks_dir, exist_ok=True)
    cap_h = cv2.VideoCapture(d_human)
    cap_o = cv2.VideoCapture(d_obj)
    T = min(int(cap_h.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT)))
    if T <= 0:
        raise RuntimeError("could not read mask frame counts")
    if osp.isfile(h5_path):
        os.remove(h5_path)
    with h5py.File(h5_path, "w") as h5:
        for i in tqdm(range(T), desc="masks -> h5"):
            rh, fh = cap_h.read()
            ro, fo = cap_o.read()
            if not rh or not ro or fh is None or fo is None:
                cap_h.release()
                cap_o.release()
                raise RuntimeError(f"mask videos ended at frame {i}/{T}")
            gh = cv2.cvtColor(fh, cv2.COLOR_BGR2GRAY) > 127
            go = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY) > 127
            h5.create_dataset(
                f"{video_prefix}/{i:06d}-k{kid}.person_mask.png",
                data=gh,
                compression="gzip",
                compression_opts=3,
            )
            h5.create_dataset(
                f"{video_prefix}/{i:06d}-k{kid}.obj_rend_mask.png",
                data=go,
                compression="gzip",
                compression_opts=3,
            )
    cap_h.release()
    cap_o.release()
    print("wrote", h5_path)

    os.makedirs(hy_sub, exist_ok=True)
    if osp.isfile(src_model):
        if osp.lexists(dst_align) or osp.isfile(dst_align):
            os.remove(dst_align)
        os.symlink(osp.relpath(src_model, hy_sub), dst_align)
        print("hy3d template symlink:", dst_align, "->", src_model)
    else:
        print("warning: object/model.obj missing; fp_hy3d_2dir will need cari4d/hy3d_staged layout")

    print(f"video_prefix={video_prefix} staged {color_out}")


if __name__ == "__main__":
    main()
