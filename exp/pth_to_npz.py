#!/usr/bin/env python3
"""
Export CoCoNet / HORefine-style ``*.pth`` (``gt`` / ``pr`` / ``in``) to experiment-folder NPZ files.

Writes only **init** (from ``in``) and **pred** (from ``pr``); **gt is not saved**.

Outputs (under ``--exp_dir``)::

  human/human_params_init.npz
  human/human_params.npz
  object/object_params_init.npz
  object/object_params.npz

Schema matches ``exp/behave_debug/Date03_Sub03_chairblack_lift_3``:

- ``human/human_params*.npz``: global_orient, body_pose, lhand_pose, rhand_pose, betas,
  trans, gender (scalar unicode), intrinsics (fx, fy, cx, cy) float64
- ``object/object_params*.npz``: angle (T, 3) float64 rotvec, trans (T, 3) float64,
  object_name scalar unicode

``intrinsics`` / ``gender`` / ``object_name`` come from ``get_intrinsics_unified`` (``--seq_name``,
``--kid``), ``--gender``, and ``--object_name`` (or inferred from ``seq_name``), not from another
experiment folder.

Example::

  python exp/pth_to_npz.py \\
    --pth output/coconet/cari4d-release+init_viz_demo/Date03_Sub03_chairblack_lift.pth \\
    --exp_dir exp/behave_debug/Date03_Sub03_chairblack_lift
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from behave_data.utils import get_intrinsics_unified  # noqa: E402


def pose_rotvec156_to_npz_fields(poses: np.ndarray):
    """Split flat SMPL-H rotvec (T, 156) like ``human_params.npz`` (see align_human2depth)."""
    poses = np.asarray(poses, dtype=np.float32)
    return (
        poses[:, :3].copy(),
        poses[:, 3:66].copy(),
        poses[:, 66:111].copy(),
        poses[:, 111:156].copy(),
    )


def infer_seq_name_from_pth(data: dict) -> str:
    for key in ("pr", "in", "gt"):
        if key not in data:
            continue
        frames = data[key].get("frames")
        if frames:
            first = frames[0]
            if isinstance(first, str) and "/" in first:
                return first.split("/")[0]
            if isinstance(first, str):
                return first
    raise ValueError("Could not infer seq_name from pth['pr|in|gt']['frames']; pass --seq_name")


def infer_object_name(seq_name: str) -> str:
    parts = seq_name.split("_")
    if len(parts) >= 3:
        return parts[2]
    return parts[-1] if parts else seq_name


def pose_abs_to_object_npz_fields(pose_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 4, 4) object pose in camera -> angle (T, 3) float64 rotvec, trans (T, 3) float64."""
    pose_abs = np.asarray(pose_abs, dtype=np.float64)
    if pose_abs.ndim != 3 or pose_abs.shape[-2:] != (4, 4):
        raise ValueError(f"pose_abs must be (T, 4, 4), got {pose_abs.shape}")
    R_mats = pose_abs[:, :3, :3]
    transl = pose_abs[:, :3, 3].astype(np.float64)
    rotvec = R.from_matrix(R_mats).as_rotvec().astype(np.float64)
    return rotvec, transl


def tensor_to_numpy_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def write_human_params_npz(
    out_path: str,
    smpl_pose: torch.Tensor,
    smpl_t: torch.Tensor,
    betas: torch.Tensor,
    *,
    gender: str,
    intrinsics: np.ndarray,
) -> None:
    poses_np = tensor_to_numpy_f32(smpl_pose)
    go, bp, lh, rh = pose_rotvec156_to_npz_fields(poses_np)
    trans = tensor_to_numpy_f32(smpl_t)
    betas_np = tensor_to_numpy_f32(betas)
    T = poses_np.shape[0]
    if trans.shape[0] != T or betas_np.shape[0] != T:
        raise ValueError(
            f"Human length mismatch: pose T={T}, trans {trans.shape}, betas {betas_np.shape}"
        )
    # Match ``setup_date03`` / reference trees: compact unicode scalars (length fits string).
    gender_arr = np.array(str(gender))
    intr = np.asarray(intrinsics, dtype=np.float64).reshape(4)
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        global_orient=go,
        body_pose=bp,
        lhand_pose=lh,
        rhand_pose=rh,
        betas=betas_np,
        trans=trans,
        gender=gender_arr,
        intrinsics=intr,
    )


def write_object_params_npz(
    out_path: str,
    pose_abs: torch.Tensor,
    object_name: str,
) -> None:
    pose_np = tensor_to_numpy_f32(pose_abs)
    angle, trans = pose_abs_to_object_npz_fields(pose_np)
    oname = np.array(str(object_name))
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, angle=angle, trans=trans, object_name=oname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pth", type=str, required=True, help="Path to run_horefine .pth")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment root (creates human/ and object/ underneath)",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default="",
        help="BEHAVE sequence name (e.g. Date03_Sub03_chairblack_lift). "
        "Default: from first entry in pth frames.",
    )
    parser.add_argument(
        "--kid",
        type=int,
        default=2,
        help="Kinect / vanilla id for get_intrinsics_unified (default: 2)",
    )
    parser.add_argument("--gender", type=str, default="male", help="SMPL-H gender (default: male)")
    parser.add_argument(
        "--object_name",
        type=str,
        default="",
        help="Short object label for object_params.npz (default: third segment of seq_name)",
    )
    args = parser.parse_args()

    pth_path = osp.abspath(args.pth)
    exp_dir = osp.abspath(args.exp_dir)
    if not osp.isfile(pth_path):
        raise FileNotFoundError(pth_path)

    raw = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "in" not in raw or "pr" not in raw:
        raise KeyError("pth must be a dict with 'in' and 'pr' keys")

    data_in = raw["in"]
    data_pr = raw["pr"]

    for k in ("smpl_pose", "smpl_t", "betas", "pose_abs"):
        if k not in data_in or k not in data_pr:
            raise KeyError(f"pth blocks must contain '{k}'")

    seq_name = args.seq_name.strip() or infer_seq_name_from_pth(raw)
    object_name = args.object_name.strip() or infer_object_name(seq_name)
    gender = args.gender
    K = get_intrinsics_unified("behave", seq_name, args.kid, wild_video=False)
    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)

    hum_dir = osp.join(exp_dir, "human")
    obj_dir = osp.join(exp_dir, "object")
    os.makedirs(hum_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)

    hi = osp.join(hum_dir, "human_params_init.npz")
    hp = osp.join(hum_dir, "human_params.npz")
    oi = osp.join(obj_dir, "object_params_init.npz")
    op = osp.join(obj_dir, "object_params.npz")

    write_human_params_npz(
        hi,
        data_in["smpl_pose"],
        data_in["smpl_t"],
        data_in["betas"],
        gender=gender,
        intrinsics=intr,
    )
    write_human_params_npz(
        hp,
        data_pr["smpl_pose"],
        data_pr["smpl_t"],
        data_pr["betas"],
        gender=gender,
        intrinsics=intr,
    )
    write_object_params_npz(oi, data_in["pose_abs"], object_name)
    write_object_params_npz(op, data_pr["pose_abs"], object_name)

    for p in (hi, hp, oi, op):
        print("wrote", p)


if __name__ == "__main__":
    main()
