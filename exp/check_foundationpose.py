#!/usr/bin/env python3
"""Export human (SMPL-H) and object meshes from FoundationPose ``*_all.pkl`` to OBJ.

The pickle holds per-frame object-to-camera poses (``fp_poses``). Human geometry comes from
the matching NLF ``*_params.pkl`` (same ``frames`` timeline).

Example::

  python exp/check_foundationpose.py \\
    --fp_pkl data/cari4d-demo/behave/fp-hy3d3-unidepth.orig/Date03_Sub03_chairblack_lift_all.pkl \\
    --out_dir output/fp_obj_check
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import joblib
import numpy as np
import torch
import trimesh

_REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lib_smpl import get_smpl  # noqa: E402


def _video_prefix_from_fp_pkl(path: str) -> str:
    base = osp.basename(path)
    if base.endswith("_all.pkl"):
        return base[: -len("_all.pkl")]
    return osp.splitext(base)[0]


def _default_model_obj(repo: str, video_prefix: str) -> str:
    candidates = [
        osp.join(repo, "experiments", "behave", f"{video_prefix}_2", "object", "model.obj"),
        osp.join(repo, "data", "cari4d-demo", "behave", "fp-hy3d3-unidepth", video_prefix, "model.obj"),
    ]
    for p in candidates:
        if osp.isfile(p):
            return p
    return candidates[0]


def _default_nlf_pkl(repo: str, video_prefix: str) -> str:
    return osp.join(
        repo,
        "data",
        "cari4d-demo",
        "behave",
        "nlf-smplh-gender-sepK",
        f"{video_prefix}_params.pkl",
    )


def _selected_kid(repo: str, video_prefix: str) -> int | None:
    jpath = osp.join(repo, "splits", "selected-views-map.json")
    if not osp.isfile(jpath):
        return None
    with open(jpath, "r", encoding="utf-8") as f:
        m = json.load(f)
    v = m.get(video_prefix)
    if v is None:
        return None
    if isinstance(v, str) and v.startswith("k") and v[1:].isdigit():
        return int(v[1:])
    if isinstance(v, (list, tuple)) and len(v) > 1:
        try:
            return int(v[1])
        except (TypeError, ValueError):
            return None
    return None


def _load_pose_streams_nlf(d: dict, kid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = np.asarray(d["poses"], dtype=np.float32)
    betas = np.asarray(d["betas"], dtype=np.float32)
    trans = np.asarray(d["transls"], dtype=np.float32)
    if poses.ndim == 3:
        if kid < 0 or kid >= poses.shape[1]:
            raise ValueError(f"nlf kid {kid} out of range for poses shape {poses.shape}")
        poses, betas, trans = poses[:, kid], betas[:, kid], trans[:, kid]
    elif poses.ndim != 2:
        raise ValueError(f"Unexpected NLF poses shape {poses.shape}")
    return poses, betas, trans


def _gender_str(d: dict) -> str:
    g = d.get("gender", "male")
    if isinstance(g, np.ndarray):
        g = g.item()
    return str(g)


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write OBJ with triangular faces; ``faces`` are 0-based indices."""
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
        for tri in faces:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {a} {b} {c}\n")


def transform_vertices(verts: np.ndarray, pose4: np.ndarray) -> np.ndarray:
    """Apply 4x4 object-to-camera transform to Nx3 vertices."""
    Rm = pose4[:3, :3].astype(np.float64)
    t = pose4[:3, 3].astype(np.float64)
    return (Rm @ verts.T).T + t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fp_pkl",
        type=str,
        default="data/cari4d-demo/behave/fp-hy3d3-unidepth.orig/Date03_Sub03_chairblack_lift_all.pkl",
        help="FoundationPose merged joblib pickle (*_all.pkl).",
    )
    p.add_argument(
        "--model_obj",
        type=str,
        default="",
        help="Object template OBJ (same mesh used for FP). Default: infer from video prefix.",
    )
    p.add_argument(
        "--nlf_pkl",
        type=str,
        default="",
        help="NLF SMPL-H params pickle (human). Default: nlf-smplh-gender-sepK/<prefix>_params.pkl.",
    )
    p.add_argument(
        "--nlf_kid",
        type=int,
        default=-1,
        help="Camera index into NLF (T,K,·). Default: -1 = use splits/selected-views-map.json or 0.",
    )
    p.add_argument(
        "--fp_kid",
        type=int,
        default=0,
        help="Camera index into fp_poses (T,K,4,4).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="output/fp_mesh_obj",
        help="Directory for written OBJ files.",
    )
    p.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 500, 1000],
        help="Frame indices (0-based) to export.",
    )
    p.add_argument("--device", type=str, default="cuda")
    a = p.parse_args()

    os.chdir(_REPO)
    fp_path = a.fp_pkl if osp.isabs(a.fp_pkl) else osp.join(_REPO, a.fp_pkl)
    if not osp.isfile(fp_path):
        sys.exit(f"Not found: {fp_path}")

    video_prefix = _video_prefix_from_fp_pkl(fp_path)
    model_obj = a.model_obj.strip()
    if not model_obj:
        model_obj = _default_model_obj(_REPO, video_prefix)
    model_obj = model_obj if osp.isabs(model_obj) else osp.join(_REPO, model_obj)
    if not osp.isfile(model_obj):
        sys.exit(f"Object template not found: {model_obj}\nPass --model_obj explicitly.")

    nlf_path = a.nlf_pkl.strip()
    if not nlf_path:
        nlf_path = _default_nlf_pkl(_REPO, video_prefix)
    nlf_path = nlf_path if osp.isabs(nlf_path) else osp.join(_REPO, nlf_path)
    if not osp.isfile(nlf_path):
        sys.exit(f"NLF pickle not found: {nlf_path}\nPass --nlf_pkl or disable human via editing script.")

    fp_data = joblib.load(fp_path)
    if "fp_poses" not in fp_data or "frames" not in fp_data:
        sys.exit(f"Unexpected fp_pkl keys: {sorted(fp_data.keys())}")
    fp_poses = np.asarray(fp_data["fp_poses"], dtype=np.float32)
    fp_frames: list[str] = list(fp_data["frames"])
    T = len(fp_frames)
    if fp_poses.shape[0] != T:
        sys.exit(f"fp_poses length {fp_poses.shape[0]} != len(frames) {T}")

    nlf_data = joblib.load(nlf_path)
    nlf_frames = list(nlf_data["frames"])
    if len(nlf_frames) != T or any(a != b for a, b in zip(fp_frames, nlf_frames)):
        print(
            "warning: FP and NLF frame lists differ; align by frame id for each index used.",
            file=sys.stderr,
        )

    nlf_kid = a.nlf_kid
    poses_arr = np.asarray(nlf_data["poses"])
    nlf_k = int(poses_arr.shape[1]) if poses_arr.ndim == 3 else 1
    if nlf_kid < 0:
        sel = _selected_kid(_REPO, video_prefix)
        nlf_kid = sel if sel is not None and sel < nlf_k else 0
    if nlf_kid >= nlf_k:
        print(
            f"warning: nlf_kid {nlf_kid} >= NLF K={nlf_k}; using kid 0",
            file=sys.stderr,
        )
        nlf_kid = 0

    poses_nlf, betas_nlf, trans_nlf = _load_pose_streams_nlf(nlf_data, nlf_kid)
    if poses_nlf.shape[0] != T:
        sys.exit(f"NLF T={poses_nlf.shape[0]} vs FP T={T}")

    mesh = trimesh.load(model_obj, process=False, force="mesh")
    v_obj = np.asarray(mesh.vertices, dtype=np.float64)
    f_obj = np.asarray(mesh.faces, dtype=np.int64)
    if f_obj.size == 0:
        sys.exit(f"No faces in object mesh: {model_obj}")

    device = a.device if torch.cuda.is_available() else "cpu"
    smpl = get_smpl(_gender_str(nlf_data), hands=True).to(device)
    faces_h = np.asarray(smpl.faces, dtype=np.int64)

    out_root = a.out_dir if osp.isabs(a.out_dir) else osp.join(_REPO, a.out_dir)
    os.makedirs(out_root, exist_ok=True)

    fp_k = min(a.fp_kid, fp_poses.shape[1] - 1)
    if a.fp_kid != fp_k:
        print(f"warning: --fp_kid {a.fp_kid} clamped to {fp_k} (K={fp_poses.shape[1]})", file=sys.stderr)

    for fi in a.frames:
        if fi < 0 or fi >= T:
            print(f"skip frame {fi}: out of range [0, {T - 1}]", file=sys.stderr)
            continue
        if fi < len(nlf_frames) and fi < len(fp_frames) and nlf_frames[fi] != fp_frames[fi]:
            print(
                f"warning: frame index {fi} id mismatch NLF {nlf_frames[fi]!r} vs FP {fp_frames[fi]!r}",
                file=sys.stderr,
            )

        pose_o = fp_poses[fi, fp_k]
        v_o = transform_vertices(v_obj, pose_o)
        stem = f"{video_prefix}_f{fi:06d}"
        write_obj(osp.join(out_root, f"{stem}_object.obj"), v_o.astype(np.float32), f_obj)

        with torch.no_grad():
            v_h = smpl(
                torch.from_numpy(poses_nlf[fi : fi + 1]).to(device),
                torch.from_numpy(betas_nlf[fi : fi + 1]).to(device),
                torch.from_numpy(trans_nlf[fi : fi + 1]).to(device),
            )[0]
        v_h_np = v_h[0].detach().cpu().numpy().astype(np.float32)
        write_obj(osp.join(out_root, f"{stem}_human.obj"), v_h_np, faces_h)

        print(f"wrote {stem}_human.obj / {stem}_object.obj  (frame id {fp_frames[fi]!r})")


if __name__ == "__main__":
    main()
