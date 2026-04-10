#!/usr/bin/env python3
"""Visualize CoCoNet tensors saved by compare_coconet_inputs.py in the same style as Trainer (708–713).

Builds the left part of train_all_*.png: stacked render/input RGB, optional mask strip (5/6-ch xyz),
stacked render/input XYZ false-color — no pose overlays / model output.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _frame(arr: np.ndarray, t: int) -> np.ndarray:
    if arr.ndim == 4:
        return arr[t]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"expected 3D or 4D array, got shape {arr.shape}")


def _chw01_to_hwc_u8(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb.transpose(1, 2, 0) * 255.0, 0.0, 255.0)
    return x.astype(np.uint8)


def _xyz3_chw_to_vis_u8(xyz3: np.ndarray) -> np.ndarray:
    return (np.clip(xyz3.transpose(1, 2, 0) + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)


def build_trainer_rgb_xyz_strip(
    render_rgb: np.ndarray,
    input_rgb: np.ndarray,
    render_xyz: np.ndarray,
    input_xyz: np.ndarray,
) -> np.ndarray:
    """Match learning/training/trainer.py visualize_rgbm + xyza/xyzb rows (no pose block)."""
    rgba = _chw01_to_hwc_u8(render_rgb)
    rgbb = _chw01_to_hwc_u8(input_rgb)
    ab = np.concatenate([rgba, rgbb], axis=0)

    c_in = input_xyz.shape[0]
    xyz_a = render_xyz[:3]
    xyz_b = input_xyz[:3]
    xyza_vis = _xyz3_chw_to_vis_u8(xyz_a)
    xyzb_vis = _xyz3_chw_to_vis_u8(xyz_b)
    xyz_stack = np.concatenate([xyza_vis, xyzb_vis], axis=0)

    if c_in == 5:
        maska = (np.clip(render_xyz[3:].transpose(1, 2, 0) * 255.0, 0.0, 255.0)).astype(np.uint8)
        maskb = (np.clip(input_xyz[3:].transpose(1, 2, 0) * 255.0, 0.0, 255.0)).astype(np.uint8)
        comb = np.concatenate(
            [
                np.concatenate([maska, np.zeros_like(maska[:, :, :1])], axis=-1),
                np.concatenate([maskb, np.zeros_like(maskb[:, :, :1])], axis=-1),
            ],
            axis=0,
        )
        comb = np.concatenate([ab, comb, xyz_stack], axis=1)
    elif c_in == 6:
        maska = (np.clip(render_xyz[3:].transpose(1, 2, 0) * 255.0, 0.0, 255.0)).astype(np.uint8)
        maskb = (np.clip(input_xyz[3:].transpose(1, 2, 0) * 255.0, 0.0, 255.0)).astype(np.uint8)
        comb_mask = np.concatenate([maska, maskb], axis=0)
        comb = np.concatenate([ab, comb_mask, xyz_stack], axis=1)
    else:
        comb = np.concatenate([ab, xyz_stack], axis=1)
    return comb


def build_input_rgb_xyz_only(input_rgb: np.ndarray, input_xyz: np.ndarray) -> np.ndarray:
    """Two rows: input RGB (uint8), input XYZ false-color (first 3 channels)."""
    top = _chw01_to_hwc_u8(input_rgb)
    bot = _xyz3_chw_to_vis_u8(input_xyz[:3])
    return np.concatenate([top, bot], axis=0)


def load_npz_dict(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--finetune_npz", type=str, default="", help="finetune_sample.npz from compare script")
    p.add_argument("--demo_npz", type=str, default="", help="demo_style_sample.npz")
    p.add_argument("--npz", type=str, action="append", default=[], help="Generic npz (repeatable); stem used as label")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--frame", type=int, default=0, help="Time index for 4D finetune tensors")
    args = p.parse_args()

    out_dir = osp.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    jobs: list[tuple[str, dict, int]] = []
    if args.finetune_npz:
        jobs.append(("finetune", load_npz_dict(args.finetune_npz), args.frame))
    if args.demo_npz:
        jobs.append(("demo_style", load_npz_dict(args.demo_npz), 0))
    for path in args.npz:
        jobs.append((osp.splitext(osp.basename(path))[0], load_npz_dict(path), args.frame))

    strips: list[tuple[str, np.ndarray]] = []

    for label, d, t in jobs:
        rr = _frame(d["render_rgbs"], t)
        ir = _frame(d["input_rgbs"], t)
        rxyz = _frame(d["render_xyz"], t)
        ixyz = _frame(d["input_xyz"], t)

        strip = build_trainer_rgb_xyz_strip(rr, ir, rxyz, ixyz)
        only = build_input_rgb_xyz_only(ir, ixyz)

        Image.fromarray(strip).save(osp.join(out_dir, f"{label}_trainer_rgb_xyz_strip_t{t}.png"))
        Image.fromarray(only).save(osp.join(out_dir, f"{label}_input_rgb_xyz_only_t{t}.png"))
        strips.append((label, strip))
        print(f"wrote {label}_trainer_rgb_xyz_strip_t{t}.png, {label}_input_rgb_xyz_only_t{t}.png")

    if len(strips) == 2:
        h = max(s.shape[0] for _, s in strips)
        cols = []
        for label, s in strips:
            if s.shape[0] < h:
                pad = np.zeros((h - s.shape[0], s.shape[1], 3), dtype=np.uint8)
                s = np.concatenate([s, pad], axis=0)
            cols.append(s)
        side = np.concatenate(cols, axis=1)
        Image.fromarray(side).save(osp.join(out_dir, "side_by_side_trainer_strip.png"))
        print("wrote side_by_side_trainer_strip.png")


if __name__ == "__main__":
    main()
