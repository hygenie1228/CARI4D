#!/usr/bin/env python3
"""Write NLF / nlf-2unidepth pickles from human/human_params_init.npz (skip NLF inference)."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import cv2
import joblib
import numpy as np

_REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def load_human_init(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    z = np.load(path, allow_pickle=True)
    poses = np.concatenate(
        [z["global_orient"], z["body_pose"], z["lhand_pose"], z["rhand_pose"]],
        axis=1,
    ).astype(np.float32)
    betas = np.asarray(z["betas"], dtype=np.float32)
    trans = np.asarray(z["trans"], dtype=np.float32)
    g = z["gender"]
    gender = str(g.item() if isinstance(g, np.ndarray) and g.shape == () else g).strip().lower()
    if gender not in ("male", "female"):
        gender = "male"
    return poses, betas, trans, gender


def staged_frame_count(exp_dir: str, video_prefix: str, kid: int) -> int:
    color = osp.join(exp_dir, "cari4d", "videos", f"{video_prefix}.{kid}.color.mp4")
    if not osp.isfile(color):
        raise FileNotFoundError(f"staged color video not found: {color}")
    cap = cv2.VideoCapture(color)
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    if n <= 0:
        meta = int(cv2.VideoCapture(color).get(cv2.CAP_PROP_FRAME_COUNT))
        if meta > 0:
            return meta
        raise RuntimeError(f"could not read frame count from {color}")
    return n


def trim_to_T(poses: np.ndarray, betas: np.ndarray, trans: np.ndarray, T: int):
    if poses.shape[0] < T:
        raise ValueError(
            f"human_params_init has {poses.shape[0]} frames but staged video has {T}"
        )
    if poses.shape[0] > T:
        print(f"trimming human init {poses.shape[0]} -> {T} frames")
    return poses[:T], betas[:T], trans[:T]


def pack_nlf(
    poses: np.ndarray,
    betas: np.ndarray,
    trans: np.ndarray,
    gender: str,
    kid: int,
) -> dict:
    T = poses.shape[0]
    frames = [f"{i:06d}" for i in range(T)]
    center = np.zeros((T, 3), dtype=np.float32)
    per_kid = {
        "poses": poses,
        "betas": betas,
        "transls": trans,
        "center_pts": center,
        "center_verts": center,
        "frames": frames,
        "gender": gender,
        "kids": [kid],
    }
    merged = {
        "poses": poses[:, None, :],
        "betas": betas[:, None, :],
        "transls": trans[:, None, :],
        "center_pts": center[:, None, :],
        "center_verts": center[:, None, :],
        "frames": frames,
        "gender": gender,
        "kids": [kid],
    }
    return per_kid, merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--human_init_npz", required=True)
    p.add_argument("--kid", type=int, default=0)
    args = p.parse_args()

    exp_dir = osp.abspath(args.exp_dir)
    init_npz = osp.abspath(args.human_init_npz)
    if not osp.isfile(init_npz):
        raise FileNotFoundError(init_npz)

    video_prefix = osp.basename(exp_dir.rstrip("/"))
    T = staged_frame_count(exp_dir, video_prefix, args.kid)
    poses, betas, trans, gender = load_human_init(init_npz)
    poses, betas, trans = trim_to_T(poses, betas, trans, T)
    per_kid, merged = pack_nlf(poses, betas, trans, gender, args.kid)

    nlf_dir = osp.join(exp_dir, "cari4d", "nlf")
    nlf_ud_dir = osp.join(exp_dir, "cari4d", "nlf-2unidepth")
    os.makedirs(nlf_dir, exist_ok=True)
    os.makedirs(nlf_ud_dir, exist_ok=True)

    paths = {
        osp.join(nlf_dir, f"{video_prefix}_params.pkl"): merged,
        osp.join(nlf_ud_dir, f"{video_prefix}_params.pkl"): merged,
        osp.join(nlf_ud_dir, f"{video_prefix}_params_k{args.kid}.pkl"): per_kid,
    }
    for path, payload in paths.items():
        joblib.dump(payload, path)
        print(f"wrote {path} (T={T}, gender={gender})")


if __name__ == "__main__":
    main()
