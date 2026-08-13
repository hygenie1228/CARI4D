#!/usr/bin/env python3
"""Overlay CoCoNet ``pth['in']`` human SMPL-H on the staged color video (debug.mp4)."""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import sys

import cv2
import imageio.v2 as imageio
import joblib
import numpy as np
import nvdiffrast.torch as dr
import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from tqdm import tqdm

_REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import Utils  # noqa: E402
from lib_smpl import get_smpl  # noqa: E402

_SMPL_UV_OBJ = osp.join(_REPO, "data/assets/smpl-meshes/meshlab-corr-order/part_surrel.obj")


def _read_nlf_gender_txt(cari4d_root: str) -> str:
    path = osp.join(cari4d_root, "nlf_gender.txt")
    if osp.isfile(path):
        with open(path, encoding="utf-8") as f:
            g = f.read().strip().lower()
        if g in ("male", "female"):
            return g
    return "male"


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


def _default_coconet_pth(exp_dir: str) -> str:
    video_prefix = osp.basename(exp_dir.rstrip("/"))
    pat = osp.join(exp_dir, "cari4d", "coconet", "**", f"{video_prefix}.pth")
    hits = sorted(glob.glob(pat, recursive=True))
    if not hits:
        raise FileNotFoundError(f"no CoCoNet pth matching {pat}")
    return hits[-1]


def _load_pth_in(pth_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "in" not in raw:
        raise KeyError(f"{pth_path} missing pth['in']")
    data = raw["in"]
    for k in ("smpl_pose", "smpl_t", "betas"):
        if k not in data:
            raise KeyError(f"pth['in'] missing {k}")
    poses = np.asarray(
        data["smpl_pose"].detach().cpu().numpy()
        if torch.is_tensor(data["smpl_pose"])
        else data["smpl_pose"],
        dtype=np.float32,
    )
    trans = np.asarray(
        data["smpl_t"].detach().cpu().numpy()
        if torch.is_tensor(data["smpl_t"])
        else data["smpl_t"],
        dtype=np.float32,
    )
    betas = np.asarray(
        data["betas"].detach().cpu().numpy()
        if torch.is_tensor(data["betas"])
        else data["betas"],
        dtype=np.float32,
    )
    return poses, trans, betas


def _load_smpl_uv_tensors(device: str) -> dict:
    verts, faces, aux = load_obj(_SMPL_UV_OBJ, load_textures=True, device=device)
    tex_maps = aux.texture_images
    if not tex_maps:
        raise RuntimeError(f"no texture map in {_SMPL_UV_OBJ}")
    verts_uvs = aux.verts_uvs.to(device)
    faces_uvs = faces.textures_idx.to(device)
    image = list(tex_maps.values())[0].to(device)[None]
    tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex)
    uv = mesh.textures.verts_uvs_padded()[0].clone()
    uv[:, 1] = 1.0 - uv[:, 1]
    return {
        "tex": mesh.textures.maps_padded().to(device).float().contiguous(),
        "uv_idx": faces.textures_idx.to(device=device, dtype=torch.int).contiguous(),
        "uv": uv.to(device).float().contiguous(),
        "pos": mesh.verts_padded()[0].to(device).float().contiguous(),
        "faces": faces.verts_idx.to(device=device, dtype=torch.int).contiguous(),
        "vnormals": mesh.verts_normals_padded()[0].to(device).float().contiguous(),
    }


def _overlay(rgb: np.ndarray, rend: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    mask = rend[..., :3].sum(axis=-1) > 1e-3
    out = rgb.astype(np.float32).copy()
    r = rend[..., :3].astype(np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * r[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_debug(
    exp_dir: str, kid: int, out_path: str, coconet_pth: str, batch: int
) -> None:
    exp_dir = osp.abspath(exp_dir)
    video_prefix = osp.basename(exp_dir.rstrip("/"))
    color_video = osp.join(exp_dir, "cari4d", "videos", f"{video_prefix}.{kid}.color.mp4")
    if not osp.isfile(color_video):
        raise FileNotFoundError(color_video)
    if not osp.isfile(coconet_pth):
        raise FileNotFoundError(coconet_pth)

    poses, trans, betas = _load_pth_in(coconet_pth)
    gender = _read_nlf_gender_txt(osp.join(exp_dir, "cari4d"))
    K = _load_intrinsics(exp_dir, kid)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for nvdiffrast debug rendering")

    smpl = get_smpl(gender, hands=True).to(device)
    mesh_tensors = _load_smpl_uv_tensors(device)
    glctx = dr.RasterizeCudaContext()

    reader = imageio.get_reader(color_video)
    fps = float(reader.get_meta_data().get("fps", 30.0) or 30.0)
    T = min(len(reader), poses.shape[0])
    if T <= 0:
        raise RuntimeError(f"no frames in {color_video}")

    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, format="FFMPEG", fps=fps)

    for start in tqdm(range(0, T, batch), desc="render pth[in] debug"):
        end = min(start + batch, T)
        with torch.no_grad():
            verts = smpl(
                torch.from_numpy(poses[start:end]).to(device),
                torch.from_numpy(betas[start:end]).to(device),
                torch.from_numpy(trans[start:end]).to(device),
            )[0].float().contiguous()
        img0 = reader.get_data(start)
        render_size = img0.shape[:2]
        color_r, _depth, _xyz = Utils.nvdiff_color_depth_render(
            [K] * (end - start), glctx, mesh_tensors, render_size, verts
        )
        color_r = (color_r.cpu().numpy() * 255.0).astype(np.uint8)

        for j, fi in enumerate(range(start, end)):
            rgb = reader.get_data(fi)
            if rgb.shape[:2] != render_size:
                rgb = cv2.resize(rgb, (render_size[1], render_size[0]), interpolation=cv2.INTER_AREA)
            panel = np.concatenate([rgb, _overlay(rgb, color_r[j])], axis=1)
            writer.append_data(panel)

    writer.close()
    print(f"wrote {out_path} ({T} frames, gender={gender}, source={coconet_pth} pth['in'])")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--kid", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--coconet_pth", default="")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    exp_dir = osp.abspath(args.exp_dir)
    coconet_pth = args.coconet_pth or _default_coconet_pth(exp_dir)
    out_path = args.out or osp.join(exp_dir, "debug.mp4")
    render_debug(exp_dir, args.kid, out_path, coconet_pth, args.batch)


if __name__ == "__main__":
    main()
