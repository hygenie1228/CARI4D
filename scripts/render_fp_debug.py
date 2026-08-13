#!/usr/bin/env python3
"""Overlay FoundationPose object tracks on the staged color video (debug.mp4).

Open4D-HOI / wild layout:
- poses from ``cari4d/fp-hy3d3-unidepth/*_all.pkl`` (``fp_poses``)
- object mesh from ``cari4d/hy3d_staged/<prefix>_export/<prefix>_align.obj``
- intrinsics from ``cari4d/intrinsics.pkl`` / ``videos/*.color.pkl``

``fp_poses`` are already ``pose @ tf_to_centered`` (see ``fp_filter_2dir.py``), so they
apply directly to the staged align mesh vertices (AABB-centered at origin).
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import cv2
import imageio.v2 as imageio
import joblib
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from tqdm import tqdm

_REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import Utils  # noqa: E402

_OBJECT_COLOR = np.array([0.15, 0.95, 0.25], dtype=np.float32)  # bright green (nvdiff 0-1)


def _load_intrinsics(exp_dir: str, kid: int) -> np.ndarray:
    cari4d = osp.join(exp_dir, "cari4d")
    video_prefix = osp.basename(exp_dir.rstrip("/"))
    for rel in (
        osp.join(cari4d, "intrinsics.pkl"),
        osp.join(cari4d, "videos", f"{video_prefix}.{kid}.color.pkl"),
    ):
        if osp.isfile(rel):
            data = joblib.load(rel)
            if "K" in data:
                return np.asarray(data["K"], dtype=np.float32)
            fx, fy, cx, cy = (float(data[k]) for k in ("fx", "fy", "cx", "cy"))
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise FileNotFoundError(f"no intrinsics under {cari4d}")


def _default_align_obj(exp_dir: str) -> str:
    video_prefix = osp.basename(exp_dir.rstrip("/"))
    return osp.join(
        exp_dir,
        "cari4d",
        "hy3d_staged",
        f"{video_prefix}_export",
        f"{video_prefix}_align.obj",
    )


def _load_object_mesh_tensors(obj_path: str, device: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Load staged align OBJ; return nvdiff tensors, template verts, and local AABB."""
    if not osp.isfile(obj_path):
        raise FileNotFoundError(obj_path)

    tm = trimesh.load(obj_path, process=False)
    if not isinstance(tm, trimesh.Trimesh):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
    verts = np.asarray(tm.vertices, dtype=np.float32)
    tm.visual.vertex_colors = np.tile(
        (np.clip(_OBJECT_COLOR, 0, 1) * 255).astype(np.uint8),
        (len(verts), 1),
    )
    mesh_tensors = Utils.make_mesh_tensors(tm, device=device)
    for k in mesh_tensors:
        if torch.is_tensor(mesh_tensors[k]):
            mesh_tensors[k] = mesh_tensors[k].contiguous()

    bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0).astype(np.float32)
    return mesh_tensors, verts, bbox


def _load_fp_poses(fp_pkl: str, kid: int) -> np.ndarray:
    d = joblib.load(fp_pkl)
    poses = np.asarray(d["fp_poses"], dtype=np.float32)
    if poses.ndim != 4:
        raise ValueError(f"expected fp_poses (T, K, 4, 4), got {poses.shape}")
    if kid < 0 or kid >= poses.shape[1]:
        raise ValueError(f"--kid {kid} out of range for fp_poses {poses.shape}")
    return poses[:, kid]


def _transform_verts(verts_base: np.ndarray, poses: np.ndarray) -> np.ndarray:
    out = np.empty((len(poses), len(verts_base), 3), dtype=np.float32)
    for i, pose in enumerate(poses):
        R, t = pose[:3, :3], pose[:3, 3]
        out[i] = verts_base @ R.T + t[None]
    return out


def _overlay(rgb: np.ndarray, rend: np.ndarray, alpha: float = 0.72) -> np.ndarray:
    mask = rend[..., :3].sum(axis=-1) > 1e-3
    out = rgb.astype(np.float32).copy()
    r = rend[..., :3].astype(np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * r[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_fp_bbox(rgb: np.ndarray, pose: np.ndarray, bbox: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Same helper style as ``fp_behave.py`` filter visualization."""
    vis = rgb.copy()
    vis = Utils.draw_posed_3d_box(K, vis, pose, bbox, line_color=(0, 255, 80), linewidth=2)
    vis = Utils.draw_xyz_axis(vis, ob_in_cam=pose, scale=0.08, K=K, thickness=2, transparency=0, is_input_rgb=True)
    return vis


def render_debug(
    exp_dir: str,
    kid: int,
    out_path: str,
    fp_pkl: str,
    obj_path: str,
    batch: int,
    draw_bbox: bool,
) -> None:
    exp_dir = osp.abspath(exp_dir)
    video_prefix = osp.basename(exp_dir.rstrip("/"))
    color_video = osp.join(exp_dir, "cari4d", "videos", f"{video_prefix}.{kid}.color.mp4")
    for p in (color_video, fp_pkl, obj_path):
        if not osp.isfile(p):
            raise FileNotFoundError(p)

    fp_poses = _load_fp_poses(fp_pkl, kid)
    K = _load_intrinsics(exp_dir, kid)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for nvdiffrast debug rendering")

    mesh_tensors, verts_base, bbox = _load_object_mesh_tensors(obj_path, device)
    glctx = dr.RasterizeCudaContext()

    reader = imageio.get_reader(color_video)
    fps = float(reader.get_meta_data().get("fps", 30.0) or 30.0)
    T = min(len(reader), fp_poses.shape[0])
    if T <= 0:
        raise RuntimeError(f"no frames in {color_video}")

    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, format="FFMPEG", fps=fps)

    for start in tqdm(range(0, T, batch), desc="render fp debug"):
        end = min(start + batch, T)
        poses_chunk = fp_poses[start:end]
        verts_np = _transform_verts(verts_base, poses_chunk)
        verts = torch.from_numpy(verts_np).to(device).float().contiguous()
        img0 = reader.get_data(start)
        render_size = img0.shape[:2]
        K_rois = [K] * (end - start)
        color_r, _depth, _xyz = Utils.nvdiff_color_depth_render(
            K_rois, glctx, mesh_tensors, render_size, verts
        )
        color_r = (color_r.cpu().numpy() * 255.0).astype(np.uint8)

        for j, fi in enumerate(range(start, end)):
            rgb = reader.get_data(fi)
            if rgb.shape[:2] != render_size:
                rgb = cv2.resize(rgb, (render_size[1], render_size[0]), interpolation=cv2.INTER_AREA)
            ov = _overlay(rgb, color_r[j])
            if draw_bbox:
                ov = _draw_fp_bbox(ov, poses_chunk[j], bbox, K)
            panel = np.concatenate([rgb, ov], axis=1)
            writer.append_data(panel)

    writer.close()
    print(f"wrote {out_path} ({T} frames, source={fp_pkl})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--kid", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--fp_pkl", default="")
    p.add_argument("--obj", default="")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--no_bbox", action="store_true", help="Skip 3D bbox / axis overlay")
    args = p.parse_args()

    exp_dir = osp.abspath(args.exp_dir)
    video_prefix = osp.basename(exp_dir.rstrip("/"))
    fp_pkl = args.fp_pkl or osp.join(
        exp_dir, "cari4d", "fp-hy3d3-unidepth", f"{video_prefix}_all.pkl"
    )
    obj_path = args.obj or _default_align_obj(exp_dir)
    out_path = args.out or osp.join(exp_dir, "debug.mp4")
    render_debug(
        exp_dir,
        args.kid,
        out_path,
        fp_pkl,
        obj_path,
        args.batch,
        draw_bbox=not args.no_bbox,
    )


if __name__ == "__main__":
    main()
