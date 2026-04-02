#!/usr/bin/env python3
"""
Create depth visualization from packed depth MP4 with noise-robust defaults.

Input is packed uint16 millimeters in BGR8 as in `run_unidepth.pack_depth_u16_to_bgr`:
  high = depth_u16 >> 8  -> B channel
  low  = depth_u16 & 0xFF -> G channel
  R channel is zero.

Pipeline (noise-robust, debug-friendly):
  1) Decode packed uint16 depth(mm) to float depth(m)
  2) Clamp invalid/out-of-range depth
  3) Downscale depth first (default: 0.5)
  4) Spatial denoise (median + gaussian)
  5) Temporal denoise (delta clip + EMA)
  6) Stabilize normalization range (lo/hi EMA) to reduce color flicker
  7) TURBO colormap + light post blur

Defaults are tuned for noisy depth videos with downscaled debug outputs:
``--downscale 0.5 --crf 24 --depth-median-ksize 5 --temporal-mix 0.45 --range-mix 0.85``.

FFmpeg: PATH, else `imageio_ffmpeg.get_ffmpeg_exe()`.

Example: # DO NOT change
    python exp/create_depth_visualization.py \
        --depth_mp4 /home/namhj/CARI4D/exp/behave_debug/Date03_Sub03_chairblack_lift_2/processed/depth.mp4 \
        --out exp/behave_debug/Date03_Sub03_chairblack_lift_2/depth_visualization.mp4 \
        --redo \
        --downscale 0.5 \
        --crf 24 \
        --debug-first-128

    python exp/create_depth_visualization.py \
        --depth_mp4 /home/namhj/CARI4D/exp/behave_debug/Date03_Sub03_chairblack_lift_3/processed/depth.back.mp4  \
        --out exp/behave_debug/Date01_Sub01_backpack_back_0/depth_visualization_new.mp4 \
        --redo \
        --downscale 0.5 \
        --crf 24 \
        --debug-first-128

"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import shutil
import subprocess

import cv2
import numpy as np
from tqdm import tqdm


def _resolve_ffmpeg_exe() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "ffmpeg not found in PATH and imageio-ffmpeg is not installed. "
            "Install system ffmpeg, or: pip install imageio-ffmpeg"
        ) from e


class _FfmpegH264VisualizationWriter:
    """H.264 + yuv420p via ffmpeg stdin; plays reliably in VS Code / browsers."""

    def __init__(
        self,
        path,
        fps,
        size,
        ffmpeg_exe: str,
        *,
        crf: int,
        preset: str,
        tune: str | None,
        pix_fmt: str,
    ):
        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid frame size: {size}")

        cmd = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            pix_fmt,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-movflags",
            "+faststart",
        ]
        if tune:
            cmd.extend(["-tune", tune])
        cmd.append(path)
        self._path = path
        self._size = size
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._released = False

    def isOpened(self):
        return self._proc is not None and self._proc.stdin is not None and self._proc.poll() is None

    def write(self, frame_bgr):
        if not self.isOpened():
            return
        if frame_bgr.shape[1] != self._size[0] or frame_bgr.shape[0] != self._size[1]:
            raise ValueError(
                f"Frame shape {frame_bgr.shape[:2]} does not match writer size {self._size[::-1]}"
            )
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        if self._released or self._proc is None:
            return
        self._released = True
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
        code = self._proc.wait()
        self._proc = None
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}) writing {self._path}: {stderr.strip()}")


def _open_visualization_writer(path, fps, size, *, crf: int, preset: str, tune: str | None, pix_fmt: str):
    exe = _resolve_ffmpeg_exe()
    label = f"libx264/ffmpeg crf={crf} preset={preset} pix_fmt={pix_fmt}"
    if tune:
        label += f" tune={tune}"
    if shutil.which("ffmpeg"):
        label += f" ({exe})"
    else:
        label += f" (imageio_ffmpeg: {exe})"
    return (
        _FfmpegH264VisualizationWriter(
            path, fps, size, exe, crf=crf, preset=preset, tune=tune, pix_fmt=pix_fmt
        ),
        label,
    )


def _maybe_warn_lossy_depth_container(depth_mp4: str) -> None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt",
                "-of",
                "default=noprint_wrappers=1:nokey=0",
                depth_mp4,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return
    text = out.stdout.lower()
    if "mpeg4" in text or "msmpeg4" in text:
        print(
            "warning: depth.mp4 looks like legacy MPEG-4 Part 2; colormap re-encode may be huge "
            "(noisy depth -> high TURBO entropy at the same CRF)."
        )
        return
    if "codec_name=h264" in text or "codec_name: h264" in text:
        if "yuv420p" in text:
            print(
                "warning: H.264 depth with yuv420p may chroma-damage packed B/G; visualization bitrate can "
                "be much higher than mp4 made from clean float depth (e.g. reference depth_visualization)."
            )


def depth_packed_bgr_to_u16_mm(frame_bgr: np.ndarray, pack_mode: str) -> np.ndarray:
    """Decode packed uint16 depth-mm from BGR frame.

    pack_mode:
      - "bg": high byte=B, low byte=G (run_unidepth packed depth.mp4)
      - "rg": high byte=R, low byte=G (many depth-reg mp4 files in this codebase)
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR frame (H,W,3), got {frame_bgr.shape}")
    b, g, r = cv2.split(frame_bgr)
    if pack_mode == "bg":
        return (b.astype(np.uint16) << 8) | g.astype(np.uint16)
    if pack_mode == "rg":
        return (r.astype(np.uint16) << 8) | g.astype(np.uint16)
    raise ValueError(f"Unknown pack mode: {pack_mode}")


def _decode_depth_m(frame_bgr: np.ndarray, pack_mode: str, max_depth_m: float) -> tuple[np.ndarray, str]:
    """Decode depth in meters; auto mode picks RG/BG by valid in-range ratio."""
    if pack_mode in ("bg", "rg"):
        d = depth_packed_bgr_to_u16_mm(frame_bgr, pack_mode).astype(np.float32) / 1000.0
        return d, pack_mode

    # auto: choose interpretation with more plausible in-range depth pixels
    d_bg = depth_packed_bgr_to_u16_mm(frame_bgr, "bg").astype(np.float32) / 1000.0
    d_rg = depth_packed_bgr_to_u16_mm(frame_bgr, "rg").astype(np.float32) / 1000.0
    v_bg = np.mean((d_bg > 0.0) & (d_bg <= max_depth_m))
    v_rg = np.mean((d_rg > 0.0) & (d_rg <= max_depth_m))
    if v_rg > v_bg:
        return d_rg, "rg"
    return d_bg, "bg"


def _resolve_pack_mode_from_path(pack_mode: str, depth_mp4: str) -> str:
    """Path-aware auto mode.

    In this project, many `*.depth-reg.mp4` files are RG-packed, while
    `processed/depth.mp4` from run_unidepth is BG-packed.
    """
    if pack_mode != "auto":
        return pack_mode
    name = osp.basename(depth_mp4).lower()
    if ".depth-reg." in name or name.endswith(".depth-reg.mp4"):
        return "rg"
    if name == "depth.mp4":
        return "bg"
    return "auto"


def depth_to_vis(
    depth_m: np.ndarray,
    lo: float,
    hi: float,
    colormap_name: str,
    invert_colormap: bool,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Meters -> colormap BGR using provided normalized range."""
    valid = (depth_m > 0) if valid_mask is None else valid_mask
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if hi <= lo:
        vis = np.zeros(depth_m.shape, dtype=np.uint8)
    else:
        norm = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
        if invert_colormap:
            norm = 1.0 - norm
        vis = (norm * 255.0).astype(np.uint8)
    colormap_table = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
    }
    vis_bgr = cv2.applyColorMap(vis, colormap_table[colormap_name])
    # Keep invalid depth explicit (black) so near-depth colors are not confused with invalid regions.
    vis_bgr[~valid] = 0
    return vis_bgr


def _blur_bgr(frame: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return frame
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def _smooth_depth(depth_m: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth noisy depth before colormap; helps reduce block artifacts and bitrate."""
    if sigma <= 0.0:
        return depth_m
    return cv2.GaussianBlur(depth_m, (0, 0), sigma)


def _sanitize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    """Set invalid values to zero and clamp far outliers."""
    if max_depth_m <= 0:
        return depth_m
    valid = (depth_m > 0.0) & (depth_m <= max_depth_m)
    out = np.zeros_like(depth_m, dtype=np.float32)
    out[valid] = depth_m[valid]
    return out


def _downscale_depth_with_mask(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    out_w: int,
    out_h: int,
    min_valid_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask-aware downscale: average only over valid pixels in each area cell."""
    valid_f = valid_mask.astype(np.float32)
    depth_weighted = depth_m * valid_f
    depth_avg = cv2.resize(depth_weighted, (out_w, out_h), interpolation=cv2.INTER_AREA)
    valid_ratio = cv2.resize(valid_f, (out_w, out_h), interpolation=cv2.INTER_AREA)
    out_valid = valid_ratio >= min_valid_ratio
    out_depth = np.zeros_like(depth_avg, dtype=np.float32)
    nz = valid_ratio > 1e-6
    out_depth[nz] = depth_avg[nz] / valid_ratio[nz]
    out_depth[~out_valid] = 0.0
    return out_depth, out_valid


def _median_depth(depth_m: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return depth_m
    if ksize % 2 == 0:
        raise ValueError("--depth-median-ksize must be odd")
    return cv2.medianBlur(depth_m.astype(np.float32), ksize)


def _clean_valid_mask(valid_mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return valid_mask
    if ksize % 2 == 0:
        raise ValueError("--valid-mask-open-ksize must be odd")
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    opened = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)


def _temporal_denoise_depth(
    current: np.ndarray,
    current_valid: np.ndarray,
    prev: np.ndarray | None,
    prev_valid: np.ndarray | None,
    alpha: float,
    clip_m: float,
) -> np.ndarray:
    """Temporal denoise with clipped update and EMA."""
    if prev is None or prev_valid is None:
        return current
    out = current.copy()
    both_valid = current_valid & prev_valid
    curr_only = current_valid & (~prev_valid)

    # Suppress one-frame spikes: do not allow abrupt depth jumps.
    if np.any(both_valid):
        delta = current[both_valid] - prev[both_valid]
        if clip_m > 0.0:
            delta = np.clip(delta, -clip_m, clip_m)
        clipped = prev[both_valid] + delta
        if alpha > 0.0:
            clipped = (1.0 - alpha) * clipped + alpha * prev[both_valid]
        out[both_valid] = clipped

    if np.any(curr_only):
        out[curr_only] = current[curr_only]
    out[~current_valid] = 0.0
    return out


def _compute_lo_hi(
    depth_m: np.ndarray,
    lo_p: float,
    hi_p: float,
    prev_lo: float | None,
    prev_hi: float | None,
    range_mix: float,
) -> tuple[float, float]:
    valid = depth_m > 0.0
    if not np.any(valid):
        if prev_lo is not None and prev_hi is not None:
            return prev_lo, prev_hi
        return 0.0, 1.0
    lo = float(np.percentile(depth_m[valid], lo_p))
    hi = float(np.percentile(depth_m[valid], hi_p))
    if prev_lo is None or prev_hi is None:
        return lo, max(hi, lo + 1e-6)
    mix = np.clip(range_mix, 0.0, 0.99)
    stab_lo = mix * prev_lo + (1.0 - mix) * lo
    stab_hi = mix * prev_hi + (1.0 - mix) * hi
    return stab_lo, max(stab_hi, stab_lo + 1e-6)


def _estimate_global_lo_hi_from_video(
    depth_mp4: str,
    *,
    lo_p: float,
    hi_p: float,
    max_depth_m: float,
    downscale: float,
    out_w: int,
    out_h: int,
    median_ksize: int,
    depth_gaussian: float,
    sample_step: int,
    max_samples: int,
    pixel_stride: int,
    pack_mode: str,
) -> tuple[float, float]:
    """Estimate one fixed colormap range from sampled reference frames."""
    if sample_step < 1:
        raise ValueError("--scale-sample-step must be >= 1")
    if max_samples < 1:
        raise ValueError("--scale-max-samples must be >= 1")
    if pixel_stride < 1:
        raise ValueError("--scale-pixel-stride must be >= 1")

    cap = cv2.VideoCapture(depth_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open scale reference video: {depth_mp4}")

    sampled = 0
    sampled_values: list[np.ndarray] = []
    idx = 0
    while sampled < max_samples:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % sample_step == 0:
            depth_m, _chosen = _decode_depth_m(frame_bgr, pack_mode, max_depth_m)
            depth_m = _sanitize_depth(depth_m, max_depth_m)
            if downscale < 1.0:
                depth_m = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_AREA)
            depth_m = _median_depth(depth_m, median_ksize)
            depth_m = _smooth_depth(depth_m, depth_gaussian)
            valid = depth_m > 0.0
            if np.any(valid):
                vals = depth_m[valid].reshape(-1)[::pixel_stride]
                if vals.size > 0:
                    sampled_values.append(vals.astype(np.float32, copy=False))
                    sampled += 1
        idx += 1

    cap.release()
    if not sampled_values:
        return 0.0, 1.0

    all_valid = np.concatenate(sampled_values, axis=0)
    lo = float(np.percentile(all_valid, lo_p))
    hi = float(np.percentile(all_valid, hi_p))
    return lo, max(hi, lo + 1e-6)


def _compute_output_size(width: int, height: int, downscale: float) -> tuple[int, int]:
    if not (0.1 <= downscale <= 1.0):
        raise ValueError("--downscale must be in [0.1, 1.0]")
    out_w = max(2, int(round(width * downscale)))
    out_h = max(2, int(round(height * downscale)))
    # yuv420p requires even dimensions.
    out_w -= out_w % 2
    out_h -= out_h % 2
    out_w = max(2, out_w)
    out_h = max(2, out_h)
    return out_w, out_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--depth_mp4",
        type=str,
        required=True,
        help="Input packed depth video: processed/depth.mp4 (run_unidepth format)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output mp4 path (default: sibling of processed/depth_visualization.mp4)",
    )
    ap.add_argument("--first_n", type=int, default=0, help="0=all frames; else first N frames")
    ap.add_argument(
        "--debug-first-128",
        action="store_true",
        help="Debug shortcut: process only the first 128 frames",
    )
    ap.add_argument("--redo", action="store_true", help="Overwrite existing output")
    ap.add_argument(
        "--crf",
        type=int,
        default=24,
        help="libx264 CRF; higher = smaller file (typical 23–36 for vis-only)",
    )
    ap.add_argument("--preset", type=str, default="slow", help="libx264 preset (veryslow = smaller, slower encode)")
    ap.add_argument(
        "--tune",
        type=str,
        default="animation",
        help="libx264 -tune (use 'none' to disable; animation often helps synthetic colormaps)",
    )
    ap.add_argument(
        "--vis-blur",
        type=float,
        default=0.6,
        help="Gaussian sigma on BGR after colormap (0=off); knocks down MPEG-4 depth noise for encoding",
    )
    ap.add_argument(
        "--temporal-mix",
        type=float,
        default=0.45,
        help="EMA blend with previous depth frame (0=off); higher value = steadier but laggier",
    )
    ap.add_argument(
        "--depth-gaussian",
        type=float,
        default=0.7,
        help="Gaussian sigma on depth(m) before colormap (0=off)",
    )
    ap.add_argument(
        "--depth-median-ksize",
        type=int,
        default=5,
        help="Median filter kernel size on depth (odd, 1=off). Strongly reduces salt-and-pepper noise.",
    )
    ap.add_argument(
        "--temporal-clip-m",
        type=float,
        default=0.12,
        help="Per-frame temporal depth jump clip in meters before EMA (0=off)",
    )
    ap.add_argument(
        "--max-depth-m",
        type=float,
        default=6.0,
        help="Clip far invalid depths above this value (meters).",
    )
    ap.add_argument(
        "--range-lo",
        type=float,
        default=2.0,
        help="Lower percentile for depth normalization on valid pixels.",
    )
    ap.add_argument(
        "--range-hi",
        type=float,
        default=98.0,
        help="Upper percentile for depth normalization on valid pixels.",
    )
    ap.add_argument(
        "--range-mix",
        type=float,
        default=0.85,
        help="EMA stabilization for percentile range across frames (0=no stabilization).",
    )
    ap.add_argument(
        "--scale-mode",
        type=str,
        default="fixed",
        choices=["fixed", "global", "adaptive"],
        help="fixed=use fixed lo/hi defaults, global=estimate once from reference video, adaptive=frame-wise lo/hi",
    )
    ap.add_argument(
        "--fixed-lo",
        type=float,
        default=0.1390,
        help="Fixed colormap lower bound in meters (used when --scale-mode fixed).",
    )
    ap.add_argument(
        "--fixed-hi",
        type=float,
        default=2.2859,
        help="Fixed colormap upper bound in meters (used when --scale-mode fixed).",
    )
    ap.add_argument(
        "--scale-ref-mp4",
        type=str,
        default=None,
        help="Reference depth mp4 used for global colormap scaling. If omitted, uses --depth_mp4.",
    )
    ap.add_argument(
        "--scale-sample-step",
        type=int,
        default=4,
        help="Sample every Nth frame for global scale estimation.",
    )
    ap.add_argument(
        "--scale-max-samples",
        type=int,
        default=160,
        help="Maximum sampled frames for global scale estimation.",
    )
    ap.add_argument(
        "--scale-pixel-stride",
        type=int,
        default=16,
        help="Use every Nth valid pixel when estimating global range.",
    )
    ap.add_argument(
        "--downscale",
        type=float,
        default=0.5,
        help="Output/depth processing scale in (0,1]. Default 0.5 for stable debug visualization.",
    )
    ap.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.35,
        help="For downscaled pixels, minimum valid coverage ratio to keep pixel as valid.",
    )
    ap.add_argument(
        "--valid-mask-open-ksize",
        type=int,
        default=3,
        help="Morphological opening kernel on valid mask (odd, 1=off). Removes isolated noisy valid speckles.",
    )
    ap.add_argument(
        "--pix-fmt",
        type=str,
        default="yuv420p",
        choices=["yuv420p", "yuv444p"],
        help="yuv420p=smaller/compatible, yuv444p=cleaner colormap but bigger files.",
    )
    ap.add_argument(
        "--pack-mode",
        type=str,
        default="auto",
        choices=["auto", "bg", "rg"],
        help="Depth byte packing: bg=(B<<8)|G, rg=(R<<8)|G, auto=choose by valid-depth ratio.",
    )
    ap.add_argument(
        "--colormap",
        type=str,
        default="jet",
        choices=["jet", "turbo", "inferno"],
        help="Color map for depth visualization. jet gives clearer near-blue/far-red mapping.",
    )
    ap.add_argument(
        "--invert-colormap",
        action="store_true",
        help="Invert depth-to-color mapping (use only if near/far appears flipped).",
    )
    args = ap.parse_args()

    tune = None if args.tune.lower() in ("none", "", "off") else args.tune
    if not (18 <= args.crf <= 40):
        raise ValueError("--crf should be between 18 and 40")
    if not (0.0 <= args.temporal_mix <= 0.95):
        raise ValueError("--temporal-mix must be in [0, 0.95]")
    if not (0.0 <= args.range_mix <= 0.99):
        raise ValueError("--range-mix must be in [0, 0.99]")
    if not (0.0 <= args.range_lo < args.range_hi <= 100.0):
        raise ValueError("--range-lo/--range-hi must satisfy 0 <= lo < hi <= 100")
    if not (args.fixed_hi > args.fixed_lo):
        raise ValueError("--fixed-hi must be greater than --fixed-lo")
    if not (0.0 <= args.min_valid_ratio <= 1.0):
        raise ValueError("--min-valid-ratio must be in [0, 1]")

    depth_mp4 = osp.abspath(args.depth_mp4)
    if not osp.isfile(depth_mp4):
        raise FileNotFoundError(depth_mp4)
    pack_mode = _resolve_pack_mode_from_path(args.pack_mode, depth_mp4)
    if args.pack_mode == "auto" and pack_mode != "auto":
        print(f"auto pack-mode by path: {pack_mode} ({osp.basename(depth_mp4)})")

    out_path = (
        osp.abspath(args.out)
        if args.out is not None
        else osp.join(osp.dirname(depth_mp4), "..", "depth_visualization.mp4")
    )

    if args.out is not None and not osp.isabs(args.out):
        out_path = osp.abspath(args.out)

    out_path = osp.normpath(out_path)

    _maybe_warn_lossy_depth_container(depth_mp4)

    if osp.exists(out_path) or osp.lexists(out_path):
        if not args.redo:
            print(f"{out_path} exists; pass --redo to overwrite")
            return
        os.remove(out_path)

    cap = cv2.VideoCapture(depth_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {depth_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ret, fr0 = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame for size")
        h, w = fr0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_w, out_h = _compute_output_size(w, h, args.downscale)
    size = (out_w, out_h)
    vis_writer, vis_codec = _open_visualization_writer(
        out_path, fps, size, crf=args.crf, preset=args.preset, tune=tune, pix_fmt=args.pix_fmt
    )
    print(
        f"depth visualization: {vis_codec}; downscale={args.downscale}, out_size={out_w}x{out_h}, "
        f"median={args.depth_median_ksize}, gaussian={args.depth_gaussian}, temporal_mix={args.temporal_mix}, "
        f"temporal_clip_m={args.temporal_clip_m}, range=({args.range_lo},{args.range_hi}), "
        f"range_mix={args.range_mix}, colormap={args.colormap}, invert={args.invert_colormap}, pack={pack_mode}"
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_first_n = args.first_n if args.first_n > 0 else 0
    if args.debug_first_128:
        effective_first_n = 128 if effective_first_n == 0 else min(effective_first_n, 128)
    if effective_first_n > 0:
        frame_count = min(frame_count, effective_first_n)
        print(f"debug frame limit enabled: first {frame_count} frames")

    global_lo: float | None = None
    global_hi: float | None = None
    if args.scale_mode == "fixed":
        global_lo, global_hi = args.fixed_lo, args.fixed_hi
        print(f"fixed colormap range: lo={global_lo:.4f}m hi={global_hi:.4f}m")
    elif args.scale_mode == "global":
        scale_ref = osp.abspath(args.scale_ref_mp4) if args.scale_ref_mp4 else depth_mp4
        if not osp.isfile(scale_ref):
            print(f"warning: scale reference not found, fallback to input depth video: {scale_ref}")
            scale_ref = depth_mp4
        global_lo, global_hi = _estimate_global_lo_hi_from_video(
            scale_ref,
            lo_p=args.range_lo,
            hi_p=args.range_hi,
            max_depth_m=args.max_depth_m,
            downscale=args.downscale,
            out_w=out_w,
            out_h=out_h,
            median_ksize=args.depth_median_ksize,
            depth_gaussian=args.depth_gaussian,
            sample_step=args.scale_sample_step,
            max_samples=args.scale_max_samples,
            pixel_stride=args.scale_pixel_stride,
            pack_mode=pack_mode,
        )
        print(
            f"global colormap range from {scale_ref}: lo={global_lo:.4f}m hi={global_hi:.4f}m "
            f"(sample_step={args.scale_sample_step}, max_samples={args.scale_max_samples})"
        )

    prev_depth_m: np.ndarray | None = None
    prev_valid_mask: np.ndarray | None = None
    prev_lo: float | None = None
    prev_hi: float | None = None
    resolved_pack_mode: str | None = None
    for _ in tqdm(range(frame_count), desc="depth visualization"):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        depth_m, chosen_pack = _decode_depth_m(frame_bgr, pack_mode, args.max_depth_m)
        if resolved_pack_mode is None:
            resolved_pack_mode = chosen_pack
            if pack_mode == "auto":
                print(f"auto pack-mode resolved to: {resolved_pack_mode}")

        depth_m = _sanitize_depth(depth_m, args.max_depth_m)
        valid_mask = depth_m > 0.0
        if args.downscale < 1.0:
            depth_m, valid_mask = _downscale_depth_with_mask(
                depth_m, valid_mask, out_w, out_h, args.min_valid_ratio
            )
        depth_m = _median_depth(depth_m, args.depth_median_ksize)
        depth_m = _smooth_depth(depth_m, args.depth_gaussian)
        valid_mask = _clean_valid_mask(valid_mask, args.valid_mask_open_ksize)
        depth_m[~valid_mask] = 0.0
        depth_m = _temporal_denoise_depth(
            depth_m, valid_mask, prev_depth_m, prev_valid_mask, args.temporal_mix, args.temporal_clip_m
        )
        depth_m[~valid_mask] = 0.0
        prev_depth_m = depth_m.copy()
        prev_valid_mask = valid_mask.copy()

        if global_lo is not None and global_hi is not None:
            lo, hi = global_lo, global_hi
        else:
            lo, hi = _compute_lo_hi(depth_m, args.range_lo, args.range_hi, prev_lo, prev_hi, args.range_mix)
            prev_lo, prev_hi = lo, hi
        vis_bgr = depth_to_vis(depth_m, lo, hi, args.colormap, args.invert_colormap, valid_mask=valid_mask)
        vis_bgr = _blur_bgr(vis_bgr, args.vis_blur)
        vis_writer.write(vis_bgr)

    vis_writer.release()
    cap.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
