#!/usr/bin/env python3
"""Render NLF *_params.pkl / *_params_k2.pkl SMPL-H meshes to an MP4 (matplotlib, headless)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import imageio.v2 as imageio
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from tqdm import tqdm  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib_smpl import get_smpl  # noqa: E402


class _PngStagingWriter:
    """Save frames as %06d.png then mux with system ffmpeg if imageio-ffmpeg is missing."""

    __slots__ = ("_dir", "_fps", "_out", "_n")

    def __init__(self, staging_dir: str, fps: float, out_path: str) -> None:
        self._dir = staging_dir
        self._fps = fps
        self._out = out_path
        self._n = 0

    def append_data(self, im: np.ndarray) -> None:
        path = os.path.join(self._dir, f"{self._n:06d}.png")
        imageio.imwrite(path, im)
        self._n += 1

    def close(self) -> None:
        if self._n == 0:
            shutil.rmtree(self._dir, ignore_errors=True)
            return
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            shutil.rmtree(self._dir, ignore_errors=True)
            raise RuntimeError(
                "imageio FFMPEG plugin missing and no ffmpeg in PATH. "
                "Install: pip install imageio-ffmpeg"
            )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(self._fps),
                "-i",
                os.path.join(self._dir, "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                self._out,
            ],
            check=True,
        )
        shutil.rmtree(self._dir, ignore_errors=True)


def _gender_str(d: dict) -> str:
    g = d.get("gender", "male")
    if isinstance(g, np.ndarray):
        g = g.item()
    return str(g)


def _load_pose_streams(d: dict, kid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = np.asarray(d["poses"], dtype=np.float32)
    betas = np.asarray(d["betas"], dtype=np.float32)
    trans = np.asarray(d["transls"], dtype=np.float32)
    if poses.ndim == 3:
        if kid < 0 or kid >= poses.shape[1]:
            raise ValueError(f"--kid {kid} out of range for poses {poses.shape}")
        poses, betas, trans = poses[:, kid], betas[:, kid], trans[:, kid]
    elif poses.ndim != 2:
        raise ValueError(f"Unexpected poses shape {poses.shape}")
    return poses, betas, trans


def _verts_all(
    smpl,
    poses: np.ndarray,
    betas: np.ndarray,
    trans: np.ndarray,
    device: str,
    batch: int,
) -> np.ndarray:
    T = poses.shape[0]
    out: list[np.ndarray] = []
    for s in tqdm(range(0, T, batch), desc="SMPL forward"):
        e = min(s + batch, T)
        with torch.no_grad():
            v = smpl(
                torch.from_numpy(poses[s:e]).to(device),
                torch.from_numpy(betas[s:e]).to(device),
                torch.from_numpy(trans[s:e]).to(device),
            )[0]
        out.append(v.cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pkl",
        type=str,
        default="data/cari4d-demo/behave/nlf-smplh-gender-sepK-2unidepth/Date03_Sub03_chairblack_debug_params_k2.pkl",
        help="NLF joblib pickle (params or params_k2).",
    )
    p.add_argument("--out", type=str, default="", help="Output .mp4 path (default: output/viz/...)")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--stride", type=int, default=2, help="Use every Nth frame (speed / size).")
    p.add_argument("--face_stride", type=int, default=3, help="Draw every Nth face (speed).")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--elev", type=float, default=15.0)
    p.add_argument("--azim_start", type=float, default=-70.0)
    p.add_argument("--azim_step", type=float, default=0.15, help="Spin azimuth per written frame (0=fixed).")
    p.add_argument("--max_frames", type=int, default=0, help="If >0, cap number of frames after stride (preview).")
    p.add_argument(
        "--kid",
        type=int,
        default=0,
        help="Camera index when poses are (T, K, D) (ignored for flat *_k2 / (T, D)).",
    )
    a = p.parse_args()

    pkl_path = a.pkl if os.path.isabs(a.pkl) else os.path.join(REPO, a.pkl)
    d = joblib.load(pkl_path)
    gender = _gender_str(d)
    poses, betas, trans = _load_pose_streams(d, a.kid)
    frame_ids = d.get("frames", [f"f{i}" for i in range(len(poses))])

    idx = list(range(0, len(poses), max(1, a.stride)))
    if a.max_frames > 0:
        idx = idx[: a.max_frames]
    poses, betas, trans = poses[idx], betas[idx], trans[idx]
    frame_ids = [frame_ids[i] for i in idx]

    device = a.device if torch.cuda.is_available() else "cpu"
    smpl = get_smpl(gender, hands=True).to(device)
    faces = np.asarray(smpl.faces, dtype=np.int64)
    faces_sub = faces[:: max(1, a.face_stride)]

    verts = _verts_all(smpl, poses, betas, trans, device, a.batch)

    if not a.out:
        stem = os.path.splitext(os.path.basename(pkl_path))[0]
        a.out = os.path.join(REPO, "output", "viz", f"nlf_smplh_preview_{stem}.mp4")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    v0 = verts - verts.reshape(-1, 3).mean(axis=0, keepdims=True)
    lim = float(np.percentile(np.abs(v0.reshape(-1, 3)), 99.0)) * 1.15 + 1e-6

    try:
        try:
            writer = imageio.get_writer(
                a.out, format="FFMPEG", fps=a.fps, codec="libx264", quality=8
            )
        except TypeError:
            writer = imageio.get_writer(a.out, format="FFMPEG", fps=a.fps)
    except ImportError:
        writer = _PngStagingWriter(
            tempfile.mkdtemp(prefix="nlf_smplh_preview_"), a.fps, a.out
        )
    plt.ioff()
    thumb_path = os.path.splitext(a.out)[0] + "_first.png"
    for fi in tqdm(range(len(verts)), desc="render frames"):
        v = verts[fi] - verts[fi].mean(axis=0, keepdims=True)
        tris = v[faces_sub]

        fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        col = Poly3DCollection(
            tris,
            facecolors=(0.65, 0.78, 0.95, 1.0),
            edgecolors=(0.15, 0.25, 0.45, 0.25),
            linewidths=0.03,
        )
        ax.add_collection3d(col)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        az = a.azim_start + fi * a.azim_step
        ax.view_init(elev=a.elev, azim=az)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{frame_ids[fi]}  ({fi + 1}/{len(verts)})", fontsize=9)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(buf)
        if fi == 0:
            imageio.imwrite(thumb_path, buf)
        plt.close(fig)

    writer.close()
    print(f"Wrote {a.out}")
    print(f"Wrote {thumb_path}")


if __name__ == "__main__":
    main()
