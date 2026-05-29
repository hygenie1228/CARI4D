#!/usr/bin/env python3
"""Build ``exp_dir/cari4d/videos/*`` and ``masks/*.h5`` for InterCap-style wild folders.

Unlike ``stage_exp_behave.py``, the experiment directory basename does **not** encode a
Kinect/view id (no trailing ``_<digits>``). The capture view index is passed explicitly
via ``--view_id`` (default ``0``), matching ``ICAP_FOCALs`` / ``ICAP_CENTERs`` in
``behave_data.const``.

Also writes ``cari4d/nlf_gender.txt`` (``male`` or ``female``) so NLF / HORefine / opt
can resolve SMPL-H gender without a BEHAVE-style subject token in the folder name.
"""
from __future__ import annotations

import argparse
import os
import shutil
import os.path as osp
import sys

import joblib
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
_scripts_dir = osp.dirname(osp.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
import stage_exp_behave as _stage_behave

write_obj_aabb_center_at_origin = _stage_behave.write_obj_aabb_center_at_origin
write_masks_h5_local_then_copy = _stage_behave.write_masks_h5_local_then_copy
from behave_data.utils import get_intrinsics_unified


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument(
        "--view_id",
        type=int,
        default=0,
        help="InterCap camera index (0–5) used in filenames and intrinsics.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate videos/masks/hy3d aligned obj even if outputs already exist.",
    )
    args = p.parse_args()
    exp_dir = osp.abspath(args.exp_dir)
    view_id = int(args.view_id)

    video_prefix = osp.basename(exp_dir.rstrip("/"))
    kid = view_id

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
    gender_path = osp.join(work_root, "nlf_gender.txt")

    if not args.force:
        core_ok = (
            osp.isfile(color_out)
            and osp.isfile(pkl_out)
            and osp.isfile(depth_out)
            and osp.isfile(h5_path)
        )
        has_src_mesh = osp.isfile(src_model)
        align_real = osp.isfile(dst_align) and not osp.islink(dst_align)
        # If we have a source OBJ, we must not skip until centered *_align.obj exists (else FP fails).
        mesh_gate = (not has_src_mesh) or align_real
        if core_ok and mesh_gate and osp.isfile(gender_path):
            print(f"skip (outputs exist): {work_root} — use --force to regenerate")
            return

    assert osp.isfile(src_color), src_color
    assert osp.isfile(d_human) and osp.isfile(d_obj), (d_human, d_obj)
    assert osp.isfile(d_bgr), d_bgr

    K = get_intrinsics_unified("intercap", video_prefix, kid, wild_video=False).astype(
        np.float32
    )
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intr_pack_path = osp.join(work_root, "intrinsics.pkl")
    os.makedirs(work_root, exist_ok=True)
    joblib.dump(
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K, "kid": int(kid)},
        intr_pack_path,
    )
    print(
        f"wrote {intr_pack_path} (intercap view_id={kid}): fx={fx} fy={fy} cx={cx} cy={cy}"
    )

    if not osp.isfile(gender_path):
        with open(gender_path, "w", encoding="utf-8") as gf:
            gf.write("male\n")
        print(f"wrote default {gender_path} (edit to female if needed)")

    os.makedirs(videos_dir, exist_ok=True)
    shutil.copy2(src_color, color_out)
    joblib.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy}, pkl_out)
    print(f"intrinsics mirrored -> {pkl_out}")

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

    print(f"video_prefix={video_prefix} view_id={kid} staged {color_out}")


if __name__ == "__main__":
    main()
