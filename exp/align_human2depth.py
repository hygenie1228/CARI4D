#!/usr/bin/env python3
"""
Align SMPL-H human parameters to depth point clouds (single-view export).

Mirrors the core logic of prep/align_nlf2unidepth.py but reads only from an
experiment folder (human_params.npz + processed depth/mask videos).

Expected layout under --exp_dir:
  human/human_params.npz   — global_orient, body_pose, lhand_pose, rhand_pose,
                             betas, trans, gender, intrinsics (fx, fy, cx, cy)
  processed/depth.mp4      — BGR uint8, depth mm packed as (R<<8)|G / 1000 -> meters
  processed/human_mask.mp4 — BGR; grayscale > 127 is human

Extra assets you must have outside exp_dir:
  SMPL-H pickles under --smplh_root (default: <repo>/data/smpl/smplh), e.g. SMPLH_male.pkl

Project imports are limited to exp.align_utils for depth filtering and ICP.
Aligned parameters are written to human/human_params_aligned.npz with the same keys
and array layout as human/human_params.npz (subset of frames if --start/--end is used).

Run from the repository root so paths resolve.

# Example (do not remove)
python exp/align_human2depth.py --exp_dir /home/namhj/CARI4D/exp/behave_debug/Date03_Sub03_chairblack_lift_3 --debug_32 --redo
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import open3d as o3d
import smplx
import torch
import trimesh
from tqdm import tqdm
try:
    from videoio import Uint16Reader
except ImportError:
    Uint16Reader = None

_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from exp.align_utils import (
    bilateral_filter_depth_cpu_fast,
    erode_depth_cpu_fast,
    translation_only_icp_torch,
)


def depth2xyzmap(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    invalid_mask = depth < 0.001
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij")
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


def bgr_frame_to_depth_meters(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr.astype(np.float32) / 1000.0
    b, g, r = cv2.split(frame_bgr)
    d16 = (r.astype(np.uint16) << 8) | g.astype(np.uint16)
    return d16.astype(np.float32) / 1000.0


def load_human_npz(path: str):
    z = np.load(path, allow_pickle=True)
    go = z["global_orient"]
    bp = z["body_pose"]
    lh = z["lhand_pose"]
    rh = z["rhand_pose"]
    poses = np.concatenate([go, bp, lh, rh], axis=1).astype(np.float32)
    betas = z["betas"].astype(np.float32)
    trans = z["trans"].astype(np.float32)
    g = z["gender"]
    gender_arr = np.array(g, copy=True)
    if isinstance(g, np.ndarray):
        gender = str(g.item()) if g.shape == () else str(g[0])
    else:
        gender = str(g)
    if gender not in ("male", "female"):
        print(f"warning: gender '{gender}' not supported by SMPL-H; using 'male'")
        gender = "male"
        gender_arr = np.array("male", dtype="<U4")
    intr = np.array(z["intrinsics"], copy=True, dtype=np.float64)
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return poses, betas, trans, gender, K, intr, gender_arr


def pose_rotvec156_to_npz_fields(poses: np.ndarray):
    """Split flat SMPL-H rotvec (T, 156) like human_params.npz components."""
    poses = np.asarray(poses, dtype=np.float32)
    return (
        poses[:, :3].copy(),
        poses[:, 3:66].copy(),
        poses[:, 66:111].copy(),
        poses[:, 111:156].copy(),
    )


def align_sequence(
    exp_dir: str,
    smplh_root: str,
    out_npz: str,
    redo: bool,
    frame_start: int,
    frame_end: int | None,
    device: str,
    vert_z_min: float,
):
    exp_dir = osp.abspath(exp_dir)
    human_npz = osp.join(exp_dir, "human", "human_params.npz")
    depth_mp4 = osp.join(exp_dir, "processed", "depth.mp4")
    mask_mp4 = osp.join(exp_dir, "processed", "human_mask.mp4")

    for p in (human_npz, depth_mp4, mask_mp4):
        if not osp.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    if osp.isfile(out_npz) and not redo:
        print(f"{out_npz} exists, use --redo to overwrite. Skipping.")
        return

    poses, betas, trans, gender, K_full, intr_save, gender_arr_save = load_human_npz(
        human_npz
    )
    T = poses.shape[0]
    scale_ratio = 2 if float(K_full[0, 2]) > 1000 else 1
    K = K_full.copy()
    K[:2, :] /= float(scale_ratio)

    end = T if frame_end is None else min(frame_end, T)
    start = max(0, frame_start)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start} end={end} (T={T})")

    cap_d = cv2.VideoCapture(depth_mp4)
    cap_m = cv2.VideoCapture(mask_mp4)
    n_d = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT))
    n_m = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_d.release()
    cap_m.release()
    if n_d != T or n_m != T:
        raise ValueError(
            f"Frame count mismatch: human_params T={T}, depth={n_d}, mask={n_m}"
        )

    body_model = smplx.create(
        model_path=smplh_root,
        model_type="smplh",
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)
    faces_np = np.asarray(body_model.faces, dtype=np.int64)

    nlf_verts_all = []
    with torch.no_grad():
        for i in range(T):
            pv = torch.from_numpy(poses[i : i + 1]).float().to(device)
            b = torch.from_numpy(betas[i : i + 1]).float().to(device)
            tr = torch.from_numpy(trans[i : i + 1]).float().to(device)
            out = body_model(
                betas=b,
                global_orient=pv[:, :3],
                body_pose=pv[:, 3:66],
                left_hand_pose=pv[:, 66:111],
                right_hand_pose=pv[:, 111:156],
                transl=tr,
                return_verts=True,
            )
            nlf_verts_all.append(out.vertices[0].detach().cpu().numpy())
    nlf_verts_all = np.stack(nlf_verts_all, axis=0)

    verts_aligned: list[np.ndarray] = []
    trans_offsets: list[np.ndarray] = []
    center_pts: list[np.ndarray] = []
    center_verts: list[np.ndarray] = []

    # Prefer Uint16Reader for BEHAVE depth-reg style videos. This avoids
    # mis-decoding when OpenCV color-converts compressed depth bytes.
    use_uint16_reader = False
    depth_reader = None
    depth_it = None
    cap_d = None
    if Uint16Reader is not None:
        try:
            depth_reader = Uint16Reader(depth_mp4)
            depth_it = iter(depth_reader)
            use_uint16_reader = True
            print("depth reader: Uint16Reader")
        except Exception:
            depth_reader = None
            use_uint16_reader = False
    if not use_uint16_reader:
        cap_d = cv2.VideoCapture(depth_mp4)
        print("depth reader: OpenCV BGR unpack fallback")

    cap_m = cv2.VideoCapture(mask_mp4)
    if frame_start > 0:
        if use_uint16_reader:
            for _ in range(frame_start):
                try:
                        _ = next(depth_it)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"Failed to seek depth reader to frame {frame_start}"
                    ) from exc
        else:
            cap_d.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        cap_m.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    iterator = range(start, end)
    for fi in tqdm(iterator, desc="align frames"):
        if use_uint16_reader:
            try:
                d_u16 = next(depth_it)
                ret_d = True
            except StopIteration:
                ret_d = False
                d_u16 = None
        else:
            ret_d, d_bgr = cap_d.read()
        ret_m, m_bgr = cap_m.read()
        if not ret_d or not ret_m:
            raise RuntimeError(f"Failed to read frame {fi} from depth/mask videos")

        if use_uint16_reader:
            depth = d_u16.astype(np.float32) / 1000.0
        else:
            depth = bgr_frame_to_depth_meters(d_bgr)
        mask_h = (cv2.cvtColor(m_bgr, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8) * 255

        h, w = depth.shape[:2]
        depth = cv2.resize(
            depth,
            (int(w / scale_ratio), int(h / scale_ratio)),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_h = cv2.resize(mask_h, (depth.shape[1], depth.shape[0]))

        depth = erode_depth_cpu_fast(depth, radius=2)
        depth = bilateral_filter_depth_cpu_fast(depth, radius=2)
        dmap_xyz = depth2xyzmap(depth, K)

        verts_nlf = nlf_verts_all[fi]
        pts_hum = dmap_xyz[mask_h > 0].reshape((-1, 3))

        if pts_hum.shape[0] < 500:
            print(
                f"warning: frame {fi}: too few depth points ({pts_hum.shape[0]}), "
                "keeping unaligned NLF vertices"
            )
            verts_aligned.append(verts_nlf)
            trans_offsets.append(np.zeros(3, dtype=np.float32))
            center_pts.append(np.zeros(3, dtype=np.float32))
            center_verts.append(np.mean(verts_nlf, axis=0).astype(np.float32))
            continue

        pts_nlf_sample = trimesh.Trimesh(verts_nlf, faces_np, process=False).sample(8000)
        z_pts_nlf = np.median(pts_nlf_sample[:, 2])
        z_pts_hum_median = np.median(pts_hum[:, 2])
        mat = np.eye(4)
        mat[:3, 3] = [0, 0, z_pts_hum_median - z_pts_nlf]
        pts_nlf = np.matmul(pts_nlf_sample, mat[:3, :3].T) + mat[:3, 3]

        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_nlf))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_hum))
        mat_icp = translation_only_icp_torch(
            src, tgt, voxel_size=0.01, max_iters=[25, 10, 5]
        )
        mat = np.matmul(mat_icp, mat)
        z_nlf_orig = np.mean(verts_nlf[:, 2])
        z_nlf_new = mat[:3, 3][2] + z_nlf_orig
        scale = z_nlf_new / (z_nlf_orig + 1e-8)
        mat_scale = np.eye(4)
        np.fill_diagonal(mat_scale, scale)
        mat_scale[3, 3] = 1
        mat_scale[:3, 3] = [0, 0, z_pts_hum_median - z_pts_nlf * scale]

        src = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                np.matmul(pts_nlf_sample, mat_scale[:3, :3].T) + mat_scale[:3, 3]
            )
        )
        mat_icp2 = translation_only_icp_torch(src, tgt, voxel_size=0.01)
        mat = np.matmul(mat_icp2, mat_scale)
        verts_nlf_align = np.matmul(verts_nlf, mat[:3, :3].T) + mat[:3, 3]

        z = np.mean(verts_nlf_align[:, 2])
        # BEHAVE prep used 1.0m; close-range exports need a much smaller floor.
        if z < vert_z_min:
            print(
                f"warning: frame {fi}: mean mesh z={z:.3f} < {vert_z_min} "
                "(likely bad ICP), reverting to raw NLF vertices"
            )
            verts_nlf_align = verts_nlf

        verts_aligned.append(verts_nlf_align)
        trans_offsets.append(np.mean(verts_nlf_align - verts_nlf, axis=0).astype(np.float32))
        center_pts.append(np.mean(pts_hum, axis=0).astype(np.float32))
        center_verts.append(np.mean(verts_nlf, axis=0).astype(np.float32))

    if use_uint16_reader:
        depth_reader.close()
    else:
        cap_d.release()
    cap_m.release()

    poses_out = poses[start:end].copy()
    betas_out = betas[start:end].copy()
    trans_out = trans[start:end].copy()
    trans_out += np.stack(trans_offsets, axis=0)
    go, bp, lh, rh = pose_rotvec156_to_npz_fields(poses_out)

    os.makedirs(osp.dirname(out_npz) or ".", exist_ok=True)
    np.savez(
        out_npz,
        global_orient=go,
        body_pose=bp,
        lhand_pose=lh,
        rhand_pose=rh,
        betas=betas_out,
        trans=trans_out,
        gender=gender_arr_save,
        intrinsics=intr_save,
    )
    print(f"Wrote aligned params (same schema as human_params.npz) to {out_npz}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_smplh = osp.join(_REPO_ROOT, "data", "smpl", "smplh")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment folder (e.g. exp/behave_debug/Date01_Sub01_backpack_back_0)",
    )
    parser.add_argument(
        "--smplh_root",
        type=str,
        default=default_smplh,
        help="Directory containing SMPLH_male.pkl / SMPLH_female.pkl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path (default: <exp_dir>/human/human_params_aligned.npz)",
    )
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="First frame index (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="Last frame exclusive; -1 = all")
    parser.add_argument(
        "--debug_32",
        action="store_true",
        help="Quick debug mode: run only the first 32 frames (equivalent to --start 0 --end 32)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--vert_z_min",
        type=float,
        default=0.03,
        help="If mean aligned vertex z (m) is below this, treat ICP as failed and keep NLF verts",
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = osp.join(osp.abspath(args.exp_dir), "human", "human_params_aligned.npz")

    start = args.start
    end = None if args.end < 0 else args.end
    if args.debug_32:
        start, end = 0, 32
        print("debug mode enabled: using first 32 frames (start=0, end=32)")

    align_sequence(
        exp_dir=args.exp_dir,
        smplh_root=osp.abspath(args.smplh_root),
        out_npz=osp.abspath(out),
        redo=args.redo,
        frame_start=start,
        frame_end=end,
        device=args.device,
        vert_z_min=args.vert_z_min,
    )


if __name__ == "__main__":
    main()
