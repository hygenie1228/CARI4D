#!/usr/bin/env python3
"""
FoundationPose + 2-directional filter using **only** files under ``exp_dir`` (no ``data/`` tree).

Required layout::

  human/human_params.npz   — intrinsics [fx,fy,cx,cy] for ``--wild_video`` *.pkl (SMPL arrays unused by FP)
  video.mp4
  processed/depth.mp4
  processed/human_mask.mp4
  processed/object_mask.mp4
  object/model.obj

``video_prefix`` for BEHAVE-style basenames is inferred from ``exp_dir`` folder name by dropping
a trailing ``_<digits>`` segment (e.g. ``.../Date03_Sub03_chairblack_lift_2`` →
``Date03_Sub03_chairblack_lift``). Use that naming convention or adjust the folder basename.

Internally uses virtual kinect **0** for staged color path / H5 keys (``...0.color.mp4``, ``_masks_k0.h5``).
FoundationPose debug dumps go under ``<exp_dir>/fp_debug/`` (not ``data/debug``).

Frame count T = min(four MP4 lengths, ``len(human_params['trans'])``).

Usage::

  python exp/run_foundationpose.py --exp_dir exp/behave_debug/Date03_Sub03_chairblack_lift_2
  python exp/run_foundationpose.py --exp_dir ... --max_frames 200
  python exp/run_foundationpose.py --exp_dir ... --max_frames 200 -fs 100
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import shutil
import sys
import traceback

import cv2
import h5py
import joblib
import numpy as np

_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Stock FP mask/tar path convention for a single exported view (virtual kinect 0).
_PACKAGED_VIEW_ID = 0

from prep.fp_behave import merge_pickles  # noqa: E402
from prep.fp_filter_2dir import FPFilterTwoDirProcessor  # noqa: E402
from prep.fp_hy3d_2dir import BehaveHy3D2DirFPRunner  # noqa: E402
from behave_data.const import START_END_FRAMES  # noqa: E402


def bgr_frame_to_depth_mm_packed(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr.astype(np.float32)
    b, g, r = cv2.split(frame_bgr)
    d16 = (r.astype(np.uint32) << 8) | g.astype(np.uint32)
    return d16.astype(np.float32)


def infer_video_prefix(exp_dir: str) -> str:
    base = osp.basename(osp.abspath(exp_dir).rstrip("/"))
    m = re.match(r"^(.+)_(\d+)$", base)
    if m:
        return m.group(1)
    return base


def _intrinsics_dict_from_npz(z: np.lib.npyio.NpzFile, path: str) -> dict[str, float]:
    if "intrinsics" not in z:
        raise KeyError(f"{path} has no intrinsics array")
    intr = np.asarray(z["intrinsics"], dtype=np.float64).reshape(-1)
    if intr.shape[0] < 4:
        raise ValueError(f"{path} intrinsics must have length >= 4, got {intr.shape}")
    return {
        "fx": float(intr[0]),
        "fy": float(intr[1]),
        "cx": float(intr[2]),
        "cy": float(intr[3]),
    }


def write_packaged_color_video_and_intrinsics_pkl(
    exp_dir: str, video_prefix: str, intr: dict, *, redo: bool
) -> str:
    """Stage ``video.mp4`` as ``videos/<prefix>.<kid>.color.mp4`` (file copy, no symlink)."""
    vdir = osp.join(exp_dir, "videos")
    os.makedirs(vdir, exist_ok=True)
    kid = _PACKAGED_VIEW_ID
    dst_color = osp.join(vdir, f"{video_prefix}.{kid}.color.mp4")
    src = osp.join(exp_dir, "video.mp4")
    if redo and osp.lexists(dst_color):
        os.remove(dst_color)
    elif osp.lexists(dst_color) and osp.islink(dst_color):
        os.remove(dst_color)
    if not osp.isfile(dst_color):
        shutil.copy2(src, dst_color)

    pkl_path = dst_color.replace(".mp4", ".pkl")
    joblib.dump(
        {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        },
        pkl_path,
    )
    return osp.abspath(dst_color)


def build_mask_h5_from_processed(
    exp_dir: str,
    seq: str,
    frame_count: int,
    redo: bool,
) -> str:
    kid = _PACKAGED_VIEW_ID
    h5_path = osp.join(exp_dir, "masks", f"{seq}_masks_k{kid}.h5")
    os.makedirs(osp.dirname(h5_path), exist_ok=True)
    if osp.isfile(h5_path) and not redo:
        try:
            with h5py.File(h5_path, "r") as h5:
                key = f"{seq}/{0:06d}-k{kid}.person_mask.png"
                if key in h5:
                    return h5_path
        except OSError:
            pass

    h_human = osp.join(exp_dir, "processed", "human_mask.mp4")
    h_obj = osp.join(exp_dir, "processed", "object_mask.mp4")
    cap_h = cv2.VideoCapture(h_human)
    cap_o = cv2.VideoCapture(h_obj)
    if not cap_h.isOpened() or not cap_o.isOpened():
        cap_h.release()
        cap_o.release()
        raise FileNotFoundError(f"Could not open mask videos:\n  {h_human}\n  {h_obj}")

    if osp.isfile(h5_path):
        os.remove(h5_path)
    with h5py.File(h5_path, "w") as h5:
        for i in range(frame_count):
            ret_h, fr_h = cap_h.read()
            ret_o, fr_o = cap_o.read()
            if not ret_h or not ret_o or fr_h is None or fr_o is None:
                cap_h.release()
                cap_o.release()
                raise RuntimeError(f"Mask videos ended before frame {i}/{frame_count}")
            gh = cv2.cvtColor(fr_h, cv2.COLOR_BGR2GRAY) > 127
            go = cv2.cvtColor(fr_o, cv2.COLOR_BGR2GRAY) > 127
            h5.create_dataset(
                f"{seq}/{i:06d}-k{kid}.person_mask.png",
                data=gh,
                compression="gzip",
                compression_opts=3,
            )
            h5.create_dataset(
                f"{seq}/{i:06d}-k{kid}.obj_rend_mask.png",
                data=go,
                compression="gzip",
                compression_opts=3,
            )
    cap_h.release()
    cap_o.release()
    print(f"wrote mask h5 ({frame_count} frames): {h5_path}")
    return h5_path


class ExpDirHy3DRunner(BehaveHy3D2DirFPRunner):
    def get_template_file(self):
        mesh = osp.join(self.args.exp_dir, "object", "model.obj")
        if not osp.isfile(mesh):
            raise FileNotFoundError(mesh)
        print("using object template from exp_dir:", mesh)
        return mesh


class ExpPackagedHy3DRunner(ExpDirHy3DRunner):
    def prepare_video_loader(self, args):
        input_color = args.video
        video_prefix = osp.basename(input_color).split(".")[0]
        output_h5_path = osp.join(args.outpath, video_prefix + "_all.pkl")
        out_dir = osp.dirname(output_h5_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.output_path = output_h5_path

        T = int(getattr(args, "_exp_packaged_frame_count"))
        self.kids = [_PACKAGED_VIEW_ID]

        class _Ctrl:
            """Minimal VideoController stand-in: FP filter reads timestamp after each frame load."""

            def __init__(self, n):
                self._n = n
                self._current_ts = 0.0

            def __len__(self):
                return self._n

            def get_current_timestamp(self):
                return self._current_ts

        self.controllers = [_Ctrl(T)]
        self.times = np.arange(0, T).tolist()
        self.video_prefix = video_prefix

        self._cap_rgb = cv2.VideoCapture(osp.join(args.exp_dir, "video.mp4"))
        self._cap_depth = cv2.VideoCapture(osp.join(args.exp_dir, "processed", "depth.mp4"))
        if not self._cap_rgb.isOpened() or not self._cap_depth.isOpened():
            raise FileNotFoundError(
                f"Could not open exp_dir video or processed/depth.mp4 under {args.exp_dir}"
            )

        slice_chunk = len(self.times) > self.get_chunk_num()
        slice_explicit = args.start != 0 or args.end != -1
        if slice_chunk or slice_explicit:
            end = len(self.times) if args.end == -1 else min(int(args.end), len(self.times))
            if end <= args.start:
                raise ValueError(
                    f"empty frame range: start={args.start} end={end} (video has {len(self.times)} frames)"
                )
            output_h5_path = osp.join(
                args.outpath, self.video_prefix + f"_{args.start:06d}-{end:06d}.pkl"
            )
            if osp.isfile(output_h5_path) and not args.redo:
                print(f"Already exists {output_h5_path}, all done")
                sys.exit(0)
            self.output_path = output_h5_path
            self.times = self.times[args.start : end]

    def load_color_depth(self, enum_idx, kids, t):
        self.controllers[enum_idx]._current_ts = float(t)
        fi = int(t)
        self._cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, fi)
        self._cap_depth.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, rgb = self._cap_rgb.read()
        ret_d, d_bgr = self._cap_depth.read()
        if not ret or not ret_d or rgb is None or d_bgr is None:
            raise RuntimeError(f"read failed at packaged frame index {fi}")
        depth_mm = bgr_frame_to_depth_mm_packed(d_bgr)
        return rgb, depth_mm.astype(np.float32)

    def process_video(self, kid_to_run, refiner=None, mesh=None, glctx=None, est=None):
        saved = START_END_FRAMES.pop(self.video_prefix, None)
        try:
            return super().process_video(kid_to_run, refiner, mesh, glctx, est)
        finally:
            if saved is not None:
                START_END_FRAMES[self.video_prefix] = saved


def _brute_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        n += 1
    cap.release()
    return n


def validate_packaged_inputs(exp_dir: str) -> tuple[int, dict[str, float]]:
    videos = {
        "video.mp4": osp.join(exp_dir, "video.mp4"),
        "processed/depth.mp4": osp.join(exp_dir, "processed", "depth.mp4"),
        "processed/human_mask.mp4": osp.join(exp_dir, "processed", "human_mask.mp4"),
        "processed/object_mask.mp4": osp.join(exp_dir, "processed", "object_mask.mp4"),
    }
    human_npz = osp.join(exp_dir, "human", "human_params.npz")
    model_obj = osp.join(exp_dir, "object", "model.obj")
    paths = {**videos, "human/human_params.npz": human_npz, "object/model.obj": model_obj}
    missing = [k for k, p in paths.items() if not osp.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing under {exp_dir}: {missing}")

    z = np.load(human_npz, allow_pickle=True)
    intr = _intrinsics_dict_from_npz(z, human_npz)
    t_npz = int(z["trans"].shape[0])

    counts = {k: _brute_frame_count(p) for k, p in videos.items()}
    t_vid = min(counts.values())
    if t_vid == 0:
        raise RuntimeError(f"Empty video(s): {counts}")
    if len(set(counts.values())) > 1:
        print(f"warning: MP4 frame counts differ {counts}")
    T = min(t_vid, t_npz)
    if T == 0:
        raise RuntimeError("zero frame count")
    if t_vid != t_npz:
        print(f"warning: min(MP4)={t_vid} vs human_params T={t_npz}; using T={T}")
    return T, intr


def get_parser() -> argparse.ArgumentParser:
    parser = FPFilterTwoDirProcessor.get_parser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment root; all inputs live under this directory only.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="If set, process at most this many frames starting at -fs/--start (caps at video length).",
    )
    parser.set_defaults(
        wild_video=True,
        data_source="behave",
        viz_path="x",
        vis_thres=0.5,
        vis_thres2=0.5,
        iou_thres2=0.3,
        angular_velo=0.1,
        occ_frames_allowed=30,
        register_png_every=10,
        register_vis_max_candidates=1,
        tstart=0.0,
        tend=None,
        kid=_PACKAGED_VIEW_ID,
    )
    return parser


def main() -> None:
    os.chdir(_REPO_ROOT)
    args = get_parser().parse_args()
    exp_dir = osp.abspath(args.exp_dir)
    args.exp_dir = exp_dir
    if not osp.isdir(exp_dir):
        sys.exit(f"Not a directory: {exp_dir}")

    video_prefix = infer_video_prefix(exp_dir)
    T, intr = validate_packaged_inputs(exp_dir)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            sys.exit("--max_frames must be positive")
        args.end = min(T, args.start + int(args.max_frames))
    args._exp_packaged_frame_count = T
    args.kid = _PACKAGED_VIEW_ID
    args.fp_debug_dir = osp.join(exp_dir, "fp_debug")

    args.video = write_packaged_color_video_and_intrinsics_pkl(
        exp_dir, video_prefix, intr, redo=bool(args.redo)
    )

    args.masks_root = osp.join(exp_dir, "masks")
    args.outpath = osp.join(exp_dir, "object")
    os.makedirs(args.outpath, exist_ok=True)

    build_mask_h5_from_processed(exp_dir, video_prefix, T, redo=bool(args.redo))

    videos = [args.video]
    range_note = f" slice=[{args.start}:{args.end})" if args.end != -1 else ""
    print(f"exp_dir={exp_dir} video_prefix={video_prefix} virtual_k={_PACKAGED_VIEW_ID} T={T}{range_note}")
    print(f"color (staged copy)={args.video}\noutpath={args.outpath}")

    try:
        for video in videos:
            args.video = video
            processor = ExpPackagedHy3DRunner(args)
            processor.process_video(_PACKAGED_VIEW_ID)
        merge_pickles(videos, args)
    except Exception:
        print(args.video, "failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
