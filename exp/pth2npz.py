#!/usr/bin/env python3
"""
Export CoCoNet / HORefine-style ``*.pth`` (``gt`` / ``pr`` / ``in``) to BEHAVE-style NPZ files.

**Default** (prediction only): with ``--exp_dir``, finds::

  <exp_dir>/cari4d/coconet/**/<video_prefix>.pth

``video_prefix`` is the directory basename without a trailing ``_<kinect_id>`` (same as
``scripts/stage_exp_behave.py``). Writes ``pth['pr']`` to::

  <exp_dir>/human/human_params.npz
  <exp_dir>/object/object_params.npz

Pass ``--pth /path/to/file.pth`` to skip discovery (any layout). Use ``--include_init`` to also
write ``human_params_init.npz`` and ``object_params_init.npz`` from ``pth['in']`` (requires an
``in`` block with the same tensor keys as ``pr``).

NPZ schema matches ``exp/behave_debug/...``:

- ``human/human_params*.npz``: global_orient, body_pose, lhand_pose, rhand_pose, betas,
  trans, gender (scalar unicode), intrinsics (fx, fy, cx, cy) float64
- ``object/object_params*.npz``: angle (T, 3) float64 rotvec, trans (T, 3) float64,
  object_name scalar unicode

``intrinsics`` / ``gender`` / ``object_name`` use ``get_intrinsics_unified`` (``--seq_name``,
``--kid``), ``--gender``, and ``--object_name`` (or inferred from pth frames / seq_name).

Kinect id: default ``--kid -1`` uses the numeric suffix on ``exp_dir`` basename (e.g.
``..._lift_2`` -> 2); if there is no suffix, kid=0. Override with ``--kid``.

Examples::

  python exp/coconet2npz.py --exp_dir experiments/behave/Date03_Sub03_chairblack_lift_2

  python exp/coconet2npz.py --exp_dir exp/behave_debug/foo \\
    --pth output/coconet/cari4d-release+init_viz_demo/Date03_Sub03_chairblack_lift.pth \\
    --kid 2 --include_init
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
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


def exp_basename_to_video_prefix_and_kid(exp_dir: str) -> tuple[str, int]:
    base = osp.basename(osp.abspath(exp_dir).rstrip("/"))
    m = re.match(r"^(.+)_(\d+)$", base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


def find_coconet_pth(exp_dir: str, video_prefix: str) -> str:
    root = osp.join(osp.abspath(exp_dir), "cari4d", "coconet")
    if not osp.isdir(root):
        raise FileNotFoundError(
            f"Expected CoCoNet directory: {root}. Pass --pth to the .pth file explicitly."
        )

    want = f"{video_prefix}.pth"
    matches: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if want in filenames:
            matches.append(osp.join(dirpath, want))

    if len(matches) == 1:
        return matches[0]
    if not matches:
        any_pth: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".pth"):
                    any_pth.append(osp.join(dirpath, fn))
        if len(any_pth) == 1:
            return any_pth[0]
        raise FileNotFoundError(
            f"No {want} under {root}; found {len(any_pth)} other .pth file(s). "
            "Pass --pth explicitly."
        )
    raise ValueError(f"Multiple {want} under {root}: {matches}")


def _require_block_keys(block: dict, keys: tuple[str, ...], label: str) -> None:
    for k in keys:
        if k not in block:
            raise KeyError(f"pth['{label}'] must contain '{k}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment root for output NPZ files (and default .pth discovery)",
    )
    parser.add_argument(
        "--pth",
        type=str,
        default="",
        help="Path to CoCoNet .pth (default: discover under exp_dir/cari4d/coconet/)",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default="",
        help="BEHAVE sequence name for intrinsics (default: from pth frames)",
    )
    parser.add_argument(
        "--kid",
        type=int,
        default=-1,
        help="Kinect id for get_intrinsics_unified; default: from exp_dir basename _<id>, else 0",
    )
    parser.add_argument("--gender", type=str, default="male", help="SMPL-H gender")
    parser.add_argument(
        "--object_name",
        type=str,
        default="",
        help="object_params object_name (default: inferred from seq_name)",
    )
    parser.add_argument(
        "--include_init",
        action="store_true",
        help="Also write *_init.npz from pth['in'] (requires 'in' block)",
    )
    args = parser.parse_args()

    exp_dir = osp.abspath(args.exp_dir)
    video_prefix, kid_default = exp_basename_to_video_prefix_and_kid(exp_dir)
    kid = kid_default if args.kid < 0 else args.kid

    if args.pth.strip():
        pth_path = osp.abspath(args.pth.strip())
        if not osp.isfile(pth_path):
            raise FileNotFoundError(pth_path)
    else:
        pth_path = find_coconet_pth(exp_dir, video_prefix)

    raw = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "pr" not in raw:
        raise KeyError("pth must be a dict with a 'pr' key")

    data_pr = raw["pr"]
    _require_block_keys(data_pr, ("smpl_pose", "smpl_t", "betas", "pose_abs"), "pr")

    seq_name = args.seq_name.strip() or infer_seq_name_from_pth(raw)
    object_name = args.object_name.strip() or infer_object_name(seq_name)
    K = get_intrinsics_unified("behave", seq_name, kid, wild_video=False)
    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)

    hum_dir = osp.join(exp_dir, "human")
    obj_dir = osp.join(exp_dir, "object")
    os.makedirs(hum_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)

    hp = osp.join(hum_dir, "human_params.npz")
    op = osp.join(obj_dir, "object_params.npz")

    write_human_params_npz(
        hp,
        data_pr["smpl_pose"],
        data_pr["smpl_t"],
        data_pr["betas"],
        gender=args.gender,
        intrinsics=intr,
    )
    write_object_params_npz(op, data_pr["pose_abs"], object_name)
    written = [hp, op]

    if args.include_init:
        if "in" not in raw:
            raise KeyError("pth must contain 'in' when using --include_init")
        data_in = raw["in"]
        _require_block_keys(data_in, ("smpl_pose", "smpl_t", "betas", "pose_abs"), "in")
        hi = osp.join(hum_dir, "human_params_init.npz")
        oi = osp.join(obj_dir, "object_params_init.npz")
        write_human_params_npz(
            hi,
            data_in["smpl_pose"],
            data_in["smpl_t"],
            data_in["betas"],
            gender=args.gender,
            intrinsics=intr,
        )
        write_object_params_npz(oi, data_in["pose_abs"], object_name)
        written.extend([hi, oi])

    print("pth:", pth_path)
    print("seq_name:", seq_name, "kid:", kid)
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
