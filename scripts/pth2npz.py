#!/usr/bin/env python3
"""
Export CoCoNet / HORefine-style ``*.pth`` (``pr`` / optional ``in``) to BEHAVE-style NPZ files.

**Never** reads ``pth['gt']`` for export (that block is dataset ground truth, not predictions).

**Default** (prediction only): with ``--exp_dir``, finds::

  <exp_dir>/cari4d/coconet/**/<video_prefix>.pth

``video_prefix`` is the directory basename without a trailing ``_<kinect_id>`` (same as
``scripts/stage_exp_behave.py``). Writes ``pth['pr']`` to::

  <exp_dir>/human/human_params.npz
  <exp_dir>/object/object_params.npz

Pass ``--pth /path/to/file.pth`` for the main checkpoint (e.g. refined ``opt``). CoCoNet baseline
NPZs (``human/human_params_coconet.npz``, ``object/object_params_coconet.npz``) are written from
``pth['pr']`` on the CoCoNet bundle (``--coconet_pth``, or auto-discovered under
``cari4d/coconet/**`` when it differs from ``--pth``).

**Pre-CoCoNet init** (``human/human_params_init.npz`` and ``object/object_params_init.npz`` when
``in`` contains ``pose_abs``) must **not** come from a refined checkpoint's ``in`` (that is
after CoCoNet). They are taken from ``pth['in']`` on the **CoCoNet** file—i.e. the input stage
before CoCoNet—using the same resolved CoCoNet path. If you only export a single CoCoNet ``.pth``
as ``--pth`` (no ``train_state`` in ``pr``), its ``in`` block is used. Optional ``--init_pth``
overrides and loads ``in`` from that file instead. Refined ``pr`` may use (T, 72) ``smpl_pose``;
expanded to 156 with ``lib_smpl.pose72to156``.

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

  python exp/pth2npz.py --exp_dir experiments/behave/Date03_Sub03_chairblack_lift_2

  python exp/pth2npz.py --exp_dir exp/behave_debug/foo \\
    --pth output/coconet/cari4d-release+init_viz_demo/Date03_Sub03_chairblack_lift.pth \\
    --kid 2
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
from lib_smpl import pose72to156  # noqa: E402


def smpl_pose_to_156_tensor(smpl_pose: torch.Tensor) -> torch.Tensor:
    """CoCoNet/refinement checkpoints may store (T,72) body+wrist or full (T,156) SMPL-H."""
    if smpl_pose.ndim != 2:
        raise ValueError(f"smpl_pose must be (T, D), got {tuple(smpl_pose.shape)}")
    d = smpl_pose.shape[-1]
    if d == 156:
        return smpl_pose
    if d == 72:
        return pose72to156(smpl_pose)
    raise ValueError(f"smpl_pose last dim must be 72 or 156, got {d}")


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
    for key in ("pr", "in"):
        if key not in data:
            continue
        frames = data[key].get("frames")
        if frames:
            first = frames[0]
            if isinstance(first, str) and "/" in first:
                return first.split("/")[0]
            if isinstance(first, str):
                return first
    raise ValueError("Could not infer seq_name from pth['pr|in']['frames']; pass --seq_name")


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
    """Prefer ``cari4d/coconet/**/{video_prefix}.pth``, else any ``cari4d/**/{video_prefix}.pth``."""
    exp_abs = osp.abspath(exp_dir)
    want = f"{video_prefix}.pth"

    def collect_under(root: str) -> list[str]:
        matches: list[str] = []
        if not osp.isdir(root):
            return matches
        for dirpath, _dirnames, filenames in os.walk(root):
            if want in filenames:
                matches.append(osp.join(dirpath, want))
        return matches

    preferred = osp.join(exp_abs, "cari4d", "coconet")
    matches = collect_under(preferred)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple {want} under {preferred}: {matches}")

    fallback_root = osp.join(exp_abs, "cari4d")
    matches = collect_under(fallback_root)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple {want} under {fallback_root}: {matches}")

    any_pth: list[str] = []
    if osp.isdir(fallback_root):
        for dirpath, _dirnames, filenames in os.walk(fallback_root):
            for fn in filenames:
                if fn.endswith(".pth"):
                    any_pth.append(osp.join(dirpath, fn))
    if len(any_pth) == 1:
        return any_pth[0]
    raise FileNotFoundError(
        f"No {want} under {fallback_root or preferred}. "
        f"Found {len(any_pth)} other .pth file(s). Pass --pth explicitly."
    )


def _require_block_keys(block: dict, keys: tuple[str, ...], label: str) -> None:
    for k in keys:
        if k not in block:
            raise KeyError(f"pth['{label}'] must contain '{k}'")


def is_refined_checkpoint_pr(pr: dict) -> bool:
    """Opt/refine checkpoints save ``train_state`` on ``pr``; raw CoCoNet bundles typically do not."""
    return "train_state" in pr


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
        help="Main .pth (refined pr or CoCoNet pr; default: discover under exp_dir/cari4d/)",
    )
    parser.add_argument(
        "--coconet_pth",
        type=str,
        default="",
        help=(
            "CoCoNet / pre-refine .pth: pr -> *_coconet.npz; in -> *_init.npz (pre-CoCoNet). "
            "If empty and --pth is refined, tries cari4d/coconet discovery when it differs from --pth."
        ),
    )
    parser.add_argument(
        "--init_pth",
        type=str,
        default="",
        help=(
            "If set, read pre-CoCoNet init from this file's pth['in'] for *_init.npz (overrides "
            "coconet file in)."
        ),
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

    pr_pose156 = smpl_pose_to_156_tensor(data_pr["smpl_pose"])
    write_human_params_npz(
        hp,
        pr_pose156,
        data_pr["smpl_t"],
        data_pr["betas"],
        gender=args.gender,
        intrinsics=intr,
    )
    write_object_params_npz(op, data_pr["pose_abs"], object_name)
    written = [hp, op]

    coconet_path = args.coconet_pth.strip()
    if coconet_path:
        coconet_path = osp.abspath(coconet_path)
        if not osp.isfile(coconet_path):
            raise FileNotFoundError(coconet_path)
    else:
        coconet_path = ""
        try:
            cand = find_coconet_pth(exp_dir, video_prefix)
            if osp.abspath(cand) != osp.abspath(pth_path):
                coconet_path = osp.abspath(cand)
        except (FileNotFoundError, ValueError):
            pass

    coco_raw: dict | None = None
    if coconet_path:
        coco_raw = torch.load(coconet_path, map_location="cpu", weights_only=False)
        if not isinstance(coco_raw, dict) or "pr" not in coco_raw:
            raise KeyError(f"coconet pth must be a dict with 'pr': {coconet_path}")
        coco_pr = coco_raw["pr"]
        _require_block_keys(coco_pr, ("smpl_pose", "smpl_t", "betas", "pose_abs"), "pr (coconet)")
        hc = osp.join(hum_dir, "human_params_coconet.npz")
        coco_pose156 = smpl_pose_to_156_tensor(coco_pr["smpl_pose"])
        write_human_params_npz(
            hc,
            coco_pose156,
            coco_pr["smpl_t"],
            coco_pr["betas"],
            gender=args.gender,
            intrinsics=intr,
        )
        written.append(hc)
        oc = osp.join(obj_dir, "object_params_coconet.npz")
        write_object_params_npz(oc, coco_pr["pose_abs"], object_name)
        written.append(oc)

    init_path = args.init_pth.strip()
    init_raw: dict | None = None
    if init_path:
        init_path = osp.abspath(init_path)
        if not osp.isfile(init_path):
            raise FileNotFoundError(init_path)
        init_raw = torch.load(init_path, map_location="cpu", weights_only=False)
    elif coco_raw is not None and "in" in coco_raw:
        init_raw = coco_raw
    elif not is_refined_checkpoint_pr(data_pr) and "in" in raw:
        init_raw = raw

    if init_raw is not None and "in" in init_raw:
        data_in = init_raw["in"]
        label = "in (pre-CoCoNet)"
        _require_block_keys(data_in, ("smpl_pose", "smpl_t", "betas"), label)
        hi = osp.join(hum_dir, "human_params_init.npz")
        in_pose156 = smpl_pose_to_156_tensor(data_in["smpl_pose"])
        write_human_params_npz(
            hi,
            in_pose156,
            data_in["smpl_t"],
            data_in["betas"],
            gender=args.gender,
            intrinsics=intr,
        )
        written.append(hi)
        if "pose_abs" in data_in:
            _require_block_keys(data_in, ("pose_abs",), label)
            oi = osp.join(obj_dir, "object_params_init.npz")
            write_object_params_npz(oi, data_in["pose_abs"], object_name)
            written.append(oi)

    print("pth:", pth_path)
    if coconet_path:
        print("coconet_pth:", coconet_path)
    if init_path:
        print("init_pth:", init_path)
    print("seq_name:", seq_name, "kid:", kid)
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
