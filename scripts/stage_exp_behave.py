#!/usr/bin/env python3
"""Build exp_dir/cari4d/videos/* and masks/*.h5 from video.mp4 + processed/*.

Output names use the kinect id from ``exp_dir`` basename: ``..._2`` ->
``<prefix>.2.color.mp4``, ``<prefix>_masks_k2.h5``, etc. (``_<digits>`` missing -> 0).
Intrinsics come from ``behave_data.utils.get_intrinsics_unified`` (BEHAVE, kid =
trailing digits of ``exp_dir`` basename, e.g. ``..._2`` -> kinect 2). Written to
``cari4d/intrinsics.pkl`` and ``videos/*.color.pkl`` for ``--wild_video`` NLF.
``human/human_params.npz`` is not read.

If ``object/model.obj`` exists, ``cari4d/hy3d_staged/.../*_align.obj`` is written as a mesh whose
axis-aligned bounding-box center sits at the origin (not a symlink to the source).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import os.path as osp
import sys
import cv2
import h5py
import joblib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from behave_data.utils import get_intrinsics_unified


def write_obj_aabb_center_at_origin(src_path: str, dst_path: str) -> np.ndarray:
    """Copy Wavefront OBJ from ``src_path`` to ``dst_path``, translating all ``v`` vertices so the
    axis-aligned bounding-box center is at the origin. Returns the AABB center subtracted (float64
    (3,)).
    """
    vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    vmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    with open(src_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "v":
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vmin[0] = min(vmin[0], x)
                vmin[1] = min(vmin[1], y)
                vmin[2] = min(vmin[2], z)
                vmax[0] = max(vmax[0], x)
                vmax[1] = max(vmax[1], y)
                vmax[2] = max(vmax[2], z)
    if not np.isfinite(vmin).all():
        raise RuntimeError(f"no vertex lines (v x y z) found in {src_path}")
    center = (vmin + vmax) * 0.5

    def _fmt(x: float) -> str:
        return format(float(x), ".9g")

    with open(src_path, encoding="utf-8", errors="replace") as fin, open(
        dst_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "v":
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x -= center[0]
                y -= center[1]
                z -= center[2]
                rest = parts[4:] if len(parts) > 4 else []
                if rest:
                    fout.write(
                        "v {} {} {} {}\n".format(
                            _fmt(x), _fmt(y), _fmt(z), " ".join(rest)
                        )
                    )
                else:
                    fout.write("v {} {} {}\n".format(_fmt(x), _fmt(y), _fmt(z)))
            else:
                fout.write(line.rstrip("\r\n") + "\n")
    return center


def write_masks_h5_local_then_copy(
    h5_path: str,
    video_prefix: str,
    kid: int,
    d_human: str,
    d_obj: str,
) -> None:
    """Write mask H5 on local disk, then copy to ``h5_path`` (avoids NFS+h5py hangs)."""
    cap_h = cv2.VideoCapture(d_human)
    cap_o = cv2.VideoCapture(d_obj)
    try:
        T = min(
            int(cap_h.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if T <= 0:
            raise RuntimeError("could not read mask frame counts")

        fd, tmp = tempfile.mkstemp(suffix=".h5", prefix="cari4d_masks_")
        os.close(fd)
        try:
            with h5py.File(tmp, "w") as h5:
                for i in tqdm(range(T), desc="masks -> h5"):
                    rh, fh = cap_h.read()
                    ro, fo = cap_o.read()
                    if not rh or not ro or fh is None or fo is None:
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
            if osp.isfile(h5_path):
                os.remove(h5_path)
            shutil.copy2(tmp, h5_path)
        finally:
            if osp.isfile(tmp):
                os.remove(tmp)
    finally:
        cap_h.release()
        cap_o.release()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate videos/masks/hy3d aligned obj even if outputs already exist.",
    )
    args = p.parse_args()
    exp_dir = osp.abspath(args.exp_dir)

    base = osp.basename(exp_dir.rstrip("/"))
    m = re.match(r"^(.+)_(\d+)$", base)
    video_prefix = m.group(1) if m else base
    kid = int(m.group(2)) if m else 0

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
        has_src_mesh = osp.isfile(src_model)
        align_real = osp.isfile(dst_align) and not osp.islink(dst_align)
        mesh_gate = (not has_src_mesh) or align_real
        if core_ok and mesh_gate:
            print(f"skip (outputs exist): {work_root} — use --force to regenerate")
            return

    assert osp.isfile(src_color), src_color
    assert osp.isfile(d_human) and osp.isfile(d_obj), (d_human, d_obj)
    assert osp.isfile(d_bgr), d_bgr

    K = get_intrinsics_unified("behave", video_prefix, kid, wild_video=False).astype(np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intr_pack_path = osp.join(work_root, "intrinsics.pkl")
    os.makedirs(work_root, exist_ok=True)
    joblib.dump(
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K, "kid": int(kid)},
        intr_pack_path,
    )
    print(
        f"wrote {intr_pack_path} (get_intrinsics_unified behave kid={kid}): "
        f"fx={fx} fy={fy} cx={cx} cy={cy}"
    )

    os.makedirs(videos_dir, exist_ok=True)
    shutil.copy2(src_color, color_out)
    joblib.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy}, pkl_out)
    print(f"intrinsics mirrored -> {pkl_out}")

    # processed/depth.mp4 from run_unidepth is already Uint16Writer / depth-reg-compatible; copy as-is.
    shutil.copy2(d_bgr, depth_out)
    print("copied", d_bgr, "->", depth_out)

    os.makedirs(masks_dir, exist_ok=True)
    write_masks_h5_local_then_copy(h5_path, video_prefix, kid, d_human, d_obj)
    print("wrote", h5_path)

    os.makedirs(hy_sub, exist_ok=True)
    if osp.isfile(src_model):
        if osp.lexists(dst_align) or osp.isfile(dst_align):
            os.remove(dst_align)
        center = write_obj_aabb_center_at_origin(src_model, dst_align)
        print(
            "hy3d template (AABB center -> origin):",
            dst_align,
            f"(shift -[{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]) from",
            src_model,
        )
    else:
        print("warning: object/model.obj missing; fp_hy3d_2dir will need cari4d/hy3d_staged layout")

    print(f"video_prefix={video_prefix} staged {color_out}")


if __name__ == "__main__":
    main()
