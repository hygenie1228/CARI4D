#!/usr/bin/env python3
"""
Compare SMPL-H meshes between aligned npz and reference k2 pkl.

- Uses the first N frames (default: 32)
- Computes mean vertex L2 distance per frame, then reports overall mean/std
- Exports frame-0 meshes as OBJ for visual inspection

"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import joblib
import numpy as np
import torch

_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from smplfitter.pt import BodyModel


def load_aligned_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    pose156 = np.concatenate(
        [z["global_orient"], z["body_pose"], z["lhand_pose"], z["rhand_pose"]], axis=1
    ).astype(np.float32)
    betas = z["betas"].astype(np.float32)
    trans = z["trans"].astype(np.float32)
    g = z["gender"]
    if isinstance(g, np.ndarray):
        gender = str(g.item()) if g.shape == () else str(g[0])
    else:
        gender = str(g)
    return pose156, betas, trans, gender


def load_ref_pkl(pkl_path: str):
    d = joblib.load(pkl_path)
    poses = np.asarray(d["poses"], dtype=np.float32)
    betas = np.asarray(d["betas"], dtype=np.float32)
    trans = np.asarray(d["transls"], dtype=np.float32)
    gender = str(d.get("gender", "male"))
    return poses, betas, trans, gender


def to_mesh_vertices(
    body_model: BodyModel, poses: np.ndarray, betas: np.ndarray, trans: np.ndarray, device: str
) -> np.ndarray:
    verts_all = []
    with torch.no_grad():
        for i in range(poses.shape[0]):
            pv = torch.from_numpy(poses[i : i + 1]).float().to(device)
            b = torch.from_numpy(betas[i : i + 1]).float().to(device)
            tr = torch.from_numpy(trans[i : i + 1]).float().to(device)
            out = body_model(pose_rotvecs=pv, shape_betas=b, trans=tr)
            verts_all.append(out["vertices"][0].cpu().numpy())
    return np.stack(verts_all, axis=0)


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        # OBJ is 1-indexed
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aligned_npz",
        type=str,
        default=(
            "exp/behave_debug/Date03_Sub03_chairblack_lift_3/"
            "human/human_params_aligned.npz"
        ),
    )
    parser.add_argument(
        "--ref_pkl",
        type=str,
        default=(
            "data/cari4d-demo/behave/nlf-smplh-gender-sepK-2unidepth/"
            "Date03_Sub03_chairblack_debug_params_k2.pkl"
        ),
    )
    parser.add_argument(
        "--smplh_root",
        type=str,
        default=osp.join(_REPO_ROOT, "data", "smpl", "smplh"),
    )
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out_obj_aligned",
        type=str,
        default=(
            "exp/behave_debug/Date03_Sub03_chairblack_lift_3/"
            "human/check_alignment_frame000_aligned.obj"
        ),
    )
    parser.add_argument(
        "--out_obj_ref",
        type=str,
        default=(
            "exp/behave_debug/Date03_Sub03_chairblack_lift_3/"
            "human/check_alignment_frame000_ref.obj"
        ),
    )
    args = parser.parse_args()

    aligned_npz = osp.abspath(args.aligned_npz)
    ref_pkl = osp.abspath(args.ref_pkl)

    poses_a, betas_a, trans_a, gender_a = load_aligned_npz(aligned_npz)
    poses_r, betas_r, trans_r, gender_r = load_ref_pkl(ref_pkl)

    T = min(args.num_frames, poses_a.shape[0], poses_r.shape[0])
    if T <= 0:
        raise ValueError("No frames available for comparison.")

    poses_a = poses_a[:T]
    betas_a = betas_a[:T]
    trans_a = trans_a[:T]
    poses_r = poses_r[:T]
    betas_r = betas_r[:T]
    trans_r = trans_r[:T]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("warning: cuda is not available, falling back to cpu")
        device = "cpu"

    # Prefer aligned gender for consistency with aligned output.
    gender = gender_a if gender_a in ("male", "female") else gender_r
    if gender not in ("male", "female"):
        print(f"warning: unsupported gender '{gender}', using 'male'")
        gender = "male"

    body_model = BodyModel("smplh", gender, model_root=osp.abspath(args.smplh_root)).to(device)
    faces = np.asarray(body_model.faces, dtype=np.int64)

    verts_a = to_mesh_vertices(body_model, poses_a, betas_a, trans_a, device=device)
    verts_r = to_mesh_vertices(body_model, poses_r, betas_r, trans_r, device=device)

    per_frame_mean = np.linalg.norm(verts_a - verts_r, axis=2).mean(axis=1)
    mean_dist = float(per_frame_mean.mean())
    std_dist = float(per_frame_mean.std())

    write_obj(osp.abspath(args.out_obj_aligned), verts_a[0], faces)
    write_obj(osp.abspath(args.out_obj_ref), verts_r[0], faces)

    print(f"Compared frames: {T}")
    print(f"Mean vertex distance over first {T} frames: {mean_dist:.6f} m")
    print(f"Std of per-frame mean distance: {std_dist:.6f} m")
    print(f"Frame-0 aligned OBJ: {osp.abspath(args.out_obj_aligned)}")
    print(f"Frame-0 reference OBJ: {osp.abspath(args.out_obj_ref)}")


if __name__ == "__main__":
    main()
