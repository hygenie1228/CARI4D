#!/usr/bin/env python3
"""Export human (SMPL-H) and object meshes from a CoCoNet / HORefine ``*.pth`` checkpoint to OBJ.

Reads ``pth['pr']`` (default) or ``pth['in']`` with: ``pose_abs`` (T,4,4), ``smpl_pose``, ``smpl_t``,
``betas``, ``frames``. Object vertices are the template ``model.obj`` transformed by ``pose_abs``.
Human mesh is SMPL-H from ``lib_smpl.get_smpl`` (same idea as ``exp/check_foundationpose.py``).

Example::

  python exp/check_coconet.py \\
    --pth output/coconet/cari4d-release+step031397_demo/Date03_Sub03_chairblack_debug.pth \\
    --model_obj experiments/behave/Date03_Sub03_chairblack_lift_2/object/model.obj \\
    --out_dir output/coconet_mesh_obj \\
    --frames 0 500 1000
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import trimesh

_REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lib_smpl import get_smpl  # noqa: E402


def _video_prefix_from_frames(frames: list) -> str:
    if not frames:
        raise ValueError("empty frames list")
    first = frames[0]
    if isinstance(first, str) and "/" in first:
        return first.split("/")[0]
    return str(first)


def _default_model_obj(repo: str, video_prefix: str) -> str:
    candidates = [
        osp.join(repo, "experiments", "behave", f"{video_prefix}_2", "object", "model.obj"),
        osp.join(repo, "data", "cari4d-demo", "behave", "fp-hy3d3-unidepth", video_prefix, "model.obj"),
    ]
    for p in candidates:
        if osp.isfile(p):
            return p
    return candidates[0]


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
        for tri in faces:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {a} {b} {c}\n")


def transform_vertices(verts: np.ndarray, pose4: np.ndarray) -> np.ndarray:
    Rm = pose4[:3, :3].astype(np.float64)
    t = pose4[:3, 3].astype(np.float64)
    return (Rm @ verts.T).T + t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pth", type=str, required=True, help="Checkpoint path (*.pth).")
    p.add_argument(
        "--block",
        type=str,
        choices=("pr", "in"),
        default="pr",
        help="Which top-level block to read (default: pr).",
    )
    p.add_argument(
        "--model_obj",
        type=str,
        default="",
        help="Object template OBJ. Default: infer from sequence name in frames.",
    )
    p.add_argument("--out_dir", type=str, default="output/coconet_mesh_obj", help="Output directory.")
    p.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 500, 1000],
        help="Frame indices (0-based) to export.",
    )
    p.add_argument("--gender", type=str, default="male", help="SMPL-H gender.")
    p.add_argument("--device", type=str, default="cuda")
    a = p.parse_args()

    os.chdir(_REPO)
    pth_path = a.pth if osp.isabs(a.pth) else osp.join(_REPO, a.pth)
    if not osp.isfile(pth_path):
        sys.exit(f"Not found: {pth_path}")

    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or a.block not in ckpt:
        sys.exit(f"*.pth must be a dict with key {a.block!r}")
    blk = ckpt[a.block]
    for key in ("pose_abs", "smpl_pose", "smpl_t", "betas", "frames"):
        if key not in blk:
            sys.exit(f"pth[{a.block!r}] missing key {key!r}")

    frames = list(blk["frames"])
    T = len(frames)
    video_prefix = _video_prefix_from_frames(frames)

    model_obj = a.model_obj.strip()
    if not model_obj:
        model_obj = _default_model_obj(_REPO, video_prefix)
    model_obj = model_obj if osp.isabs(model_obj) else osp.join(_REPO, model_obj)
    if not osp.isfile(model_obj):
        sys.exit(f"Object template not found: {model_obj}\nPass --model_obj explicitly.")

    pa = np.asarray(blk["pose_abs"].detach().cpu().numpy(), dtype=np.float32)
    if pa.shape != (T, 4, 4):
        sys.exit(f"pose_abs shape {pa.shape}, expected ({T}, 4, 4)")

    poses = np.asarray(blk["smpl_pose"].detach().cpu().numpy(), dtype=np.float32)
    trans = np.asarray(blk["smpl_t"].detach().cpu().numpy(), dtype=np.float32)
    betas = np.asarray(blk["betas"].detach().cpu().numpy(), dtype=np.float32)
    if poses.shape[0] != T or trans.shape[0] != T or betas.shape[0] != T:
        sys.exit(
            f"length mismatch: pose {poses.shape[0]}, trans {trans.shape[0]}, betas {betas.shape[0]}, T={T}"
        )

    mesh = trimesh.load(model_obj, process=False, force="mesh")
    v_obj = np.asarray(mesh.vertices, dtype=np.float64)
    f_obj = np.asarray(mesh.faces, dtype=np.int64)
    if f_obj.size == 0:
        sys.exit(f"No faces in object mesh: {model_obj}")

    device = a.device if torch.cuda.is_available() else "cpu"
    smpl = get_smpl(str(a.gender), hands=True).to(device)
    faces_h = np.asarray(smpl.faces, dtype=np.int64)

    out_root = a.out_dir if osp.isabs(a.out_dir) else osp.join(_REPO, a.out_dir)
    os.makedirs(out_root, exist_ok=True)

    for fi in a.frames:
        if fi < 0 or fi >= T:
            print(f"skip frame {fi}: out of range [0, {T - 1}]", file=sys.stderr)
            continue

        pose_o = pa[fi]
        v_o = transform_vertices(v_obj, pose_o)
        stem = f"{video_prefix}_f{fi:06d}"
        if a.block != "pr":
            stem = f"{stem}_{a.block}"
        write_obj(osp.join(out_root, f"{stem}_object.obj"), v_o.astype(np.float32), f_obj)

        with torch.no_grad():
            v_h = smpl(
                torch.from_numpy(poses[fi : fi + 1]).to(device),
                torch.from_numpy(betas[fi : fi + 1]).to(device),
                torch.from_numpy(trans[fi : fi + 1]).to(device),
            )[0]
        v_h_np = v_h[0].detach().cpu().numpy().astype(np.float32)
        write_obj(osp.join(out_root, f"{stem}_human.obj"), v_h_np, faces_h)

        print(f"wrote {stem}_human.obj / {stem}_object.obj  (frame id {frames[fi]!r})")


if __name__ == "__main__":
    main()
