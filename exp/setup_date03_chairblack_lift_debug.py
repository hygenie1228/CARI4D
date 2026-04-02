#!/usr/bin/env python3
"""
Build exp/behave_debug/Date03_Sub03_chairblack_lift_2 from data/cari4d-demo/behave
(demo.sh video: Date03_Sub03_chairblack_lift.2.color.mp4), matching the layout of
exp/behave_debug/Date01_Sub01_backpack_back_0 where assets exist under the demo tree.

Requires the *cari4d* conda env (h5py, cv2, numpy, tqdm, joblib, videoio).

  source ~/miniconda3/etc/profile.d/conda.sh   # adjust if your conda is elsewhere
  source activate cari4d

  cd /path/to/CARI4D
  python exp/setup_date03_chairblack_lift_debug.py

Then you can run:

  python exp/align_human2depth.py --exp_dir exp/behave_debug/Date03_Sub03_chairblack_lift_2

What gets written
-----------------
  video.mp4                  — RGB, one frame per NLF-aligned time (from .color.mp4)
  processed/depth.mp4        — uint16 depth (mm) packed into BGR for align_human2depth
  processed/human_mask.mp4   — from *_masks_k{k}.h5 person_mask
  processed/object_mask.mp4  — from *_masks_k{k}.h5 obj_rend_mask
  human/human_params.npz     — from nlf-smplh-gender-sepK-2unidepth *_params_k{k}.pkl
  object/model.obj           — Hy3D/RGBA mesh .obj from data/cari4d-demo/meshes (if present)

Optional demo artifacts (not in Date01 but copied when found):
  mesh_visualization.mp4     — fp-hy3d3-unidepth filter preview, if present.

Not produced here (same as many minimal trees): gs_*, object_params.npz, depth_visualization.mp4.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import subprocess
import shutil
import sys

import cv2
import h5py
import joblib
import numpy as np
from tqdm import tqdm
from videoio import Uint16Reader

_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from behave_data.utils import get_intrinsics_unified


def depth_mm_u16_to_align_bgr(depth_mm: np.ndarray) -> np.ndarray:
    """align_human2depth.bgr_frame_to_depth_meters: low byte in G, high byte in R."""
    d = depth_mm.astype(np.uint16)
    g = (d & 0xFF).astype(np.uint8)
    r = ((d >> 8) & 0xFF).astype(np.uint8)
    b = np.zeros_like(g, dtype=np.uint8)
    return cv2.merge([b, g, r])


def mask_u8_to_bgr(mask_hw: np.ndarray) -> np.ndarray:
    m = (mask_hw.astype(np.uint8) * 255) if mask_hw.max() <= 1 else mask_hw.astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def closest_frame_index(times_s: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times_s - t)))


def open_mp4_writer(path: str, fps: float, wh: tuple[int, int], fourcc_str: str = "mp4v"):
    w, h = wh
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed for {path} (fourcc={fourcc_str})")
    return out


def get_video_codec_name(path: str) -> str | None:
    """
    Returns codec_name for the first video stream, or None if ffprobe fails.
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            stderr=subprocess.STDOUT,
        )
        s = out.decode("utf-8", errors="ignore").strip()
        return s or None
    except Exception:
        return None


def ensure_h264(in_path: str, out_path: str, crf: int = 18) -> None:
    """
    Ensure `out_path` is an H.264(yuv420p) MP4 readable by VSCode.
    If `in_path` is already H.264, we move it to `out_path`.
    """
    same_path = osp.abspath(in_path) == osp.abspath(out_path)
    codec = get_video_codec_name(in_path)
    if codec == "h264":
        if in_path != out_path:
            if osp.exists(out_path) or osp.lexists(out_path):
                os.remove(out_path)
            shutil.move(in_path, out_path)
        return

    # Re-encode with ffmpeg to guarantee codec compatibility.
    # Note: -an because these generated videos are video-only.
    tmp_out = out_path + ".tmp_h264.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        in_path,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        tmp_out,
    ]
    subprocess.run(cmd, check=True)
    if osp.exists(out_path) or osp.lexists(out_path):
        os.remove(out_path)
    os.replace(tmp_out, out_path)
    # Remove input only after successful encode.
    if osp.exists(in_path) and not same_path:
        os.remove(in_path)


def pkl_to_human_params_npz(pkl: dict, kid: int, seq_name: str, out_npz: str):
    poses = np.asarray(pkl["poses"], dtype=np.float32)
    betas = np.asarray(pkl["betas"], dtype=np.float32)
    trans = np.asarray(pkl["transls"], dtype=np.float32)

    kids = pkl.get("kids", None)
    # For some variants, pkl stores only one view (K=1). We must not index axis=1 by
    # the raw kinect id; instead we map via `kids`.
    kid_axis_idx: int | None = None
    if kids is not None and isinstance(kids, list):
        if kid in kids:
            kid_axis_idx = kids.index(kid)
        elif len(kids) == 1:
            kid_axis_idx = 0

    if poses.ndim == 3:
        if kid_axis_idx is None:
            # Fallback: if K==1, take it; otherwise error.
            kid_axis_idx = 0 if poses.shape[1] == 1 else kid
        poses = poses[:, kid_axis_idx, :]
    if betas.ndim == 3:
        if kid_axis_idx is None:
            kid_axis_idx = 0 if betas.shape[1] == 1 else kid
        betas = betas[:, kid_axis_idx, :]
    if trans.ndim == 3:
        if kid_axis_idx is None:
            kid_axis_idx = 0 if trans.shape[1] == 1 else kid
        trans = trans[:, kid_axis_idx, :]

    go = poses[:, :3]
    bp = poses[:, 3:66]
    lh = poses[:, 66:111]
    rh = poses[:, 111:156]

    gender = pkl.get("gender", "male")
    if not isinstance(gender, str):
        gender = str(gender)
    gender_arr = np.array(gender, dtype="<U8")

    K = get_intrinsics_unified("behave", seq_name, kid, wild_video=False)
    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)

    os.makedirs(osp.dirname(out_npz) or ".", exist_ok=True)
    np.savez(
        out_npz,
        global_orient=go,
        body_pose=bp,
        lhand_pose=lh,
        rhand_pose=rh,
        betas=betas,
        trans=trans,
        gender=gender_arr,
        intrinsics=intr,
    )


def iter_sequential_frames_capture(cap: cv2.VideoCapture, idx_list: list[int]):
    """Yield BGR frames for cap, assuming idx_list is non-decreasing."""
    current = -1
    fr = None
    for want in idx_list:
        while current < want:
            ret, fr = cap.read()
            if not ret or fr is None:
                raise RuntimeError(f"VideoCapture ended before frame index {want}")
            current += 1
        yield fr


def iter_sequential_uint16(reader: Uint16Reader, idx_list: list[int]):
    current = -1
    fr = None
    it = iter(reader)
    for want in idx_list:
        while current < want:
            fr = next(it)
            current += 1
        yield fr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo_root",
        type=str,
        default=_REPO_ROOT,
        help="Repository root (default: parent of exp/)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Experiment folder (default: <repo>/exp/behave_debug/Date03_Sub03_chairblack_lift_2)",
    )
    parser.add_argument(
        "--demo_behave",
        type=str,
        default=None,
        help="data/cari4d-demo/behave (default: <repo>/data/cari4d-demo/behave)",
    )
    parser.add_argument(
        "--meshes_root",
        type=str,
        default=None,
        help="Hy3D meshes root (default: <repo>/data/cari4d-demo/meshes)",
    )
    parser.add_argument("--kid", type=int, default=2, help="Kinect / camera id (demo.sh uses 2)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs under out_dir",
    )
    parser.add_argument(
        "--symlink_video",
        action="store_true",
        help="If set, skip re-encoding RGB video (only for debugging; breaks frame/T match).",
    )
    args = parser.parse_args()

    repo = osp.abspath(args.repo_root)
    demo = osp.abspath(args.demo_behave or osp.join(repo, "data", "cari4d-demo", "behave"))
    meshes_root = osp.abspath(args.meshes_root or osp.join(repo, "data", "cari4d-demo", "meshes"))
    out_dir = osp.abspath(
        args.out_dir or osp.join(repo, "exp", "behave_debug", "Date03_Sub03_chairblack_lift_2")
    )

    color_mp4 = osp.join(demo, "videos", "Date03_Sub03_chairblack_lift.2.color.mp4")
    depth_mp4 = osp.join(demo, "videos", "Date03_Sub03_chairblack_lift.2.depth-reg.mp4")
    time_json = osp.join(demo, "videos", "Date03_Sub03_chairblack_lift.2.time.json")
    h5_path = osp.join(demo, "masks", "Date03_Sub03_chairblack_lift_masks_k2.h5")
    video_prefix = "Date03_Sub03_chairblack_lift"
    # Use UNALIGNED NLF (do not run align_nlf2unidepth).
    # We'll align it later with exp/align_human2depth.py (ICP).
    nlf_pkl = osp.join(
        demo,
        "nlf-smplh-gender-sepK",
        f"{video_prefix}_params.pkl",
    )

    for p in (color_mp4, depth_mp4, time_json, h5_path, nlf_pkl):
        if not osp.isfile(p):
            raise FileNotFoundError(f"Missing demo asset: {p}")

    if not args.overwrite and osp.isdir(out_dir) and os.listdir(out_dir):
        print(
            f"{out_dir} is non-empty; refusing to run without --overwrite "
            "(delete the folder or pass --overwrite)."
        )
        sys.exit(1)

    # Clean up leftover debug intermediates from previous runs.
    # (User request: do not generate `raw.mp4` artifacts.)
    if args.overwrite and osp.isdir(out_dir):
        for p in glob.glob(osp.join(out_dir, "**", "*.raw.mp4"), recursive=True):
            try:
                os.remove(p)
            except OSError:
                pass

    os.makedirs(out_dir, exist_ok=True)
    proc_dir = osp.join(out_dir, "processed")
    hum_dir = osp.join(out_dir, "human")
    obj_dir = osp.join(out_dir, "object")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(hum_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)

    pkl = joblib.load(nlf_pkl)
    frames: list[str] = list(pkl["frames"])
    T = len(frames)
    times_s = np.array([float(s[1:]) for s in frames], dtype=np.float64)

    with open(time_json, "r", encoding="utf-8") as f:
        times_meta = json.load(f)
    color_times = np.array(times_meta["color"], dtype=np.float64) / 1e6
    depth_times = np.array(times_meta["depth"], dtype=np.float64) / 1e6

    color_idx = [closest_frame_index(color_times, t) for t in times_s]
    depth_idx = [closest_frame_index(depth_times, t) for t in times_s]

    # One-time shape probe (OpenCV: H, W = rows, cols)
    cap0 = cv2.VideoCapture(color_mp4)
    ret, rgb0 = cap0.read()
    cap0.release()
    if not ret:
        raise RuntimeError(f"Could not read {color_mp4}")
    h0, w0 = rgb0.shape[:2]
    vid_wh = (w0, h0)  # VideoWriter (width, height)
    if (w0, h0) != (2048, 1536):
        print(f"warning: expected 2048x1536 RGB, got {w0}x{h0}")

    human_npz_out = osp.join(hum_dir, "human_params.npz")
    pkl_to_human_params_npz(pkl, kid=args.kid, seq_name=video_prefix, out_npz=human_npz_out)
    print(f"Wrote {human_npz_out}")

    video_out = osp.join(out_dir, "video.mp4")
    video_out_src = osp.join(out_dir, "video.mp4.tmp_src.mp4")
    depth_out = osp.join(proc_dir, "depth.mp4")
    hmask_out = osp.join(proc_dir, "human_mask.mp4")
    hmask_out_src = osp.join(proc_dir, "human_mask.mp4.tmp_src.mp4")
    omask_out = osp.join(proc_dir, "object_mask.mp4")
    omask_out_src = osp.join(proc_dir, "object_mask.mp4.tmp_src.mp4")

    if args.symlink_video:
        if osp.lexists(video_out):
            os.remove(video_out)
        os.symlink(osp.relpath(color_mp4, out_dir), video_out)
        print(f"Symlinked {video_out} -> {color_mp4} (length may not match T={T})")
    else:
        # Write with OpenCV, then (if needed) re-encode to H.264 via ffmpeg.
        if osp.exists(video_out_src):
            os.remove(video_out_src)
        w_rgb = open_mp4_writer(video_out_src, args.fps, vid_wh)
        cap_rgb = cv2.VideoCapture(color_mp4)
        try:
            for fr in tqdm(
                iter_sequential_frames_capture(cap_rgb, color_idx),
                total=T,
                desc="video.mp4",
            ):
                w_rgb.write(fr)
        finally:
            cap_rgb.release()
            w_rgb.release()
        ensure_h264(video_out_src, video_out)
        print(f"Wrote {video_out}")

    # Depth:
    #   Write packed uint16 depth bytes via BGR8 frames, then encode with ffmpeg
    #   using libx264rgb (CRF=0) so the packed bytes survive roundtrip exactly.
    # NOTE:
    # We store depth as packed uint16 bytes encoded into BGR8:
    #   low byte -> G, high byte -> R, B is 0.
    # Using OpenCV's mp4v writer corrupts these bytes (lossy chroma/subsampling),
    # which then breaks ICP alignment downstream.
    # So we first write per-frame PNG (lossless), then encode the final mp4 with
    # libx264rgb (CRF 0) using ffmpeg to preserve the packed bytes.
    depth_frames_dir = osp.join(proc_dir, "depth_frames_bgr24")
    if osp.exists(depth_frames_dir):
        shutil.rmtree(depth_frames_dir)
    os.makedirs(depth_frames_dir, exist_ok=True)
    # Write masks then ensure H.264.
    if osp.exists(hmask_out_src):
        os.remove(hmask_out_src)
    if osp.exists(omask_out_src):
        os.remove(omask_out_src)
    w_hm = open_mp4_writer(hmask_out_src, args.fps, vid_wh)
    w_om = open_mp4_writer(omask_out_src, args.fps, vid_wh)

    h5 = h5py.File(h5_path, "r")
    try:
        depth_reader = Uint16Reader(depth_mp4)
        try:
            for frame_idx, (ft, d_fr) in enumerate(
                tqdm(
                    zip(frames, iter_sequential_uint16(depth_reader, depth_idx)),
                    total=T,
                    desc="depth(frames) + masks",
                )
            ):
                depth_bgr = depth_mm_u16_to_align_bgr(d_fr)
                depth_png_path = osp.join(depth_frames_dir, f"{frame_idx:06d}.png")
                ok = cv2.imwrite(depth_png_path, depth_bgr)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite failed for {depth_png_path}")
                mh = h5[f"{video_prefix}/{ft}-k{args.kid}.person_mask.png"][:]
                mo = h5[f"{video_prefix}/{ft}-k{args.kid}.obj_rend_mask.png"][:]
                w_hm.write(mask_u8_to_bgr(mh))
                w_om.write(mask_u8_to_bgr(mo))
        finally:
            depth_reader.close()
    finally:
        h5.close()

    w_hm.release()
    w_om.release()
    # Encode the packed depth frames into mp4 while preserving BGR bytes.
    # We pick a conservative preset to avoid any unintended color conversions.
    # (CRF=0 + qp=0 => effectively lossless for the pixel values)
    depth_encode_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-framerate",
        str(args.fps),
        "-i",
        osp.join(depth_frames_dir, "%06d.png"),
        "-an",
        "-c:v",
        "libx264rgb",
        "-crf",
        "0",
        "-qp",
        "0",
        "-preset",
        "veryslow",
        "-pix_fmt",
        "bgr24",
        "-movflags",
        "+faststart",
        depth_out,
    ]
    subprocess.run(depth_encode_cmd, check=True)
    # Cleanup intermediate frames
    shutil.rmtree(depth_frames_dir)
    print(f"Wrote {depth_out} (lossless packed depth); now fixing masks to H.264")

    # Re-encode masks if necessary (VSCode compatibility).
    ensure_h264(hmask_out_src, hmask_out)
    ensure_h264(omask_out_src, omask_out)
    print(f"Wrote {hmask_out}, {omask_out}")

    mesh_glob = osp.join(meshes_root, "Date03_Sub03_chairblack_lift*_k2_rgba_align.obj")
    meshes = sorted(glob.glob(mesh_glob))
    if meshes:
        shutil.copy2(meshes[0], osp.join(obj_dir, "model.obj"))
        print(f"Copied mesh to {obj_dir}/model.obj from {meshes[0]}")
    else:
        print(f"warning: no mesh matching {mesh_glob}; skipped object/model.obj")

    fp_vis = osp.join(
        demo,
        "fp-hy3d3-unidepth",
        "Date03_Sub03_chairblack_lift_000000-001420_k2_filter_k2.mp4",
    )
    if osp.isfile(fp_vis):
        mesh_vis_out = osp.join(out_dir, "mesh_visualization.mp4")
        # If the source is not readable by ffmpeg, skip to avoid leaving a broken file.
        try:
            if osp.exists(mesh_vis_out) or osp.lexists(mesh_vis_out):
                os.remove(mesh_vis_out)
            ensure_h264(fp_vis, mesh_vis_out)
            print(f"Wrote mesh_visualization.mp4 (H.264)")
        except Exception as e:
            print(f"warning: failed to encode mesh_visualization.mp4: {e} (skipped)")

    print("Done. Next: python exp/align_human2depth.py --exp_dir", out_dir)


if __name__ == "__main__":
    main()