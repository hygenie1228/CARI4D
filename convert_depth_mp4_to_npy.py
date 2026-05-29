#!/usr/bin/env python3
"""Decode CARI4D / BEHAVE-style depth video to NumPy with minimal round-trip loss.

``processed/depth.mp4`` (from ``exp/run_unidepth.py``) is written with
``videoio.Uint16Writer``: each frame is **uint16 depth in millimeters** in the same
container format as ``*.depth-reg.mp4`` used by ``behave_data.video_reader``.

This script **prefers ``videoio.Uint16Reader``** so the decode path matches
``ColorDepthController`` / ``prep/align_monodmap.py`` / ``exp/align_human2depth.py``.

If ``Uint16Reader`` cannot open the file (wrong build, corrupt file), it falls back to
OpenCV + packed BGR decode:

- ``bg``: ``depth_mm = (B << 8) | G`` — typical for ``processed/depth.mp4`` from run_unidepth
  (see ``exp/create_depth_visualization._resolve_pack_mode_from_path``).
- ``rg``: ``depth_mm = (R << 8) | G`` — common for many ``*.depth-reg.mp4`` assets.

Examples::

  python convert_depth_mp4_to_npy.py \\
    --input experiments/behave/Date03_Sub03_backpack_back_0/processed/depth.mp4

  python convert_depth_mp4_to_npy.py -i path/to/seq.0.depth-reg.mp4 --pack_mode rg
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

try:
    from videoio import Uint16Reader
except ImportError:
    Uint16Reader = None  # type: ignore[misc, assignment]


def _depth_packed_bgr_to_u16_mm(frame_bgr: np.ndarray, pack_mode: str) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR (H,W,3), got {frame_bgr.shape}")
    b, g, r = cv2.split(frame_bgr)
    if pack_mode == "bg":
        return (b.astype(np.uint16) << 8) | g.astype(np.uint16)
    if pack_mode == "rg":
        return (r.astype(np.uint16) << 8) | g.astype(np.uint16)
    raise ValueError(f"pack_mode must be bg or rg, got {pack_mode!r}")


def _default_pack_mode(path: str) -> str:
    name = os.path.basename(path).lower()
    if ".depth-reg." in name or name.endswith(".depth-reg.mp4"):
        return "rg"
    if name == "depth.mp4":
        return "bg"
    return "bg"


def _read_uint16_reader(path: str) -> tuple[np.ndarray, str]:
    if Uint16Reader is None:
        raise RuntimeError("videoio.Uint16Reader not installed")
    reader = Uint16Reader(path)
    try:
        n = int(reader.video_params.get("length", 0) or 0)
        frames: list[np.ndarray] = []
        it = iter(reader)
        pbar = tqdm(total=n if n > 0 else None, desc="Uint16Reader")
        while True:
            try:
                d = next(it)
            except StopIteration:
                break
            frames.append(np.asarray(d, dtype=np.uint16))
            pbar.update(1)
        pbar.close()
        if not frames:
            raise RuntimeError(f"No frames read from {path!r}")
        out = np.stack(frames, axis=0)
        return out, "uint16_reader"
    finally:
        if hasattr(reader, "close"):
            reader.close()


def _read_opencv_packed(path: str, pack_mode: str) -> tuple[np.ndarray, str]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open {path!r}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    for _ in tqdm(range(n), desc=f"OpenCV+{pack_mode}"):
        ret, bgr = cap.read()
        if not ret or bgr is None:
            break
        frames.append(_depth_packed_bgr_to_u16_mm(bgr, pack_mode))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path!r} (OpenCV path)")
    out = np.stack(frames, axis=0)
    return out, f"opencv_bgr_{pack_mode}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to depth.mp4 or *.depth-reg.mp4",
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help="Output .npy path for uint16 mm stack (default: <input_stem>_depth_u16.npy)",
    )
    p.add_argument(
        "--also_meters",
        action="store_true",
        help="Also write float32 meters array (*_depth_m.npy), d = u16 / 1000",
    )
    p.add_argument(
        "--pack_mode",
        choices=("auto", "bg", "rg"),
        default="auto",
        help="OpenCV fallback only: BGR packing (auto uses filename heuristics)",
    )
    p.add_argument(
        "--prefer_opencv",
        action="store_true",
        help="Skip Uint16Reader and use OpenCV+packed decode (debug / compare only)",
    )
    args = p.parse_args()

    inp = os.path.abspath(args.input)
    if not os.path.isfile(inp):
        print(f"Missing file: {inp}", file=sys.stderr)
        sys.exit(1)

    pack = _default_pack_mode(inp) if args.pack_mode == "auto" else args.pack_mode

    meta: dict = {"input": inp, "decode": "", "shape": [], "dtype": ""}

    if args.prefer_opencv:
        depth_u16, tag = _read_opencv_packed(inp, pack)
    else:
        try:
            depth_u16, tag = _read_uint16_reader(inp)
        except Exception as e:
            print(f"Uint16Reader failed ({e!r}); falling back to OpenCV + pack_mode={pack!r}")
            depth_u16, tag = _read_opencv_packed(inp, pack)

    meta["decode"] = tag
    meta["shape"] = list(depth_u16.shape)
    meta["dtype"] = str(depth_u16.dtype)

    out_u16 = args.output.strip() or inp.rsplit(".", 1)[0] + "_depth_u16.npy"
    out_u16 = os.path.abspath(out_u16)
    os.makedirs(os.path.dirname(out_u16) or ".", exist_ok=True)
    np.save(out_u16, depth_u16)
    print(f"Wrote {out_u16} shape={depth_u16.shape} dtype={depth_u16.dtype} decode={tag}")

    if args.also_meters:
        out_dir, fname = os.path.split(out_u16)
        stem, _ext = os.path.splitext(fname)
        if stem.endswith("_depth_u16"):
            stem_m = stem[: -len("_depth_u16")] + "_depth_m"
        else:
            stem_m = stem + "_m"
        out_m = os.path.join(out_dir, stem_m + ".npy")
        d_m = depth_u16.astype(np.float32) / 1000.0
        np.save(out_m, d_m)
        print(f"Wrote {out_m} shape={d_m.shape} dtype={d_m.dtype}")

    meta_path = out_u16[:-4] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
