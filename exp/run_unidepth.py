import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
UNIDEPTH_DIR = osp.join(ROOT_DIR, "thirdparties", "UniDepth")
if UNIDEPTH_DIR not in sys.path:
    sys.path.insert(0, UNIDEPTH_DIR)

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole

try:
    from videoio import Uint16Writer
except ImportError as exc:
    Uint16Writer = None
    _VIDEOIO_IMPORT_ERROR = exc
else:
    _VIDEOIO_IMPORT_ERROR = None


def _open_writer(path, fps, size):
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Failed to open VideoWriter for: {path}")


class _FfmpegH264VisualizationWriter:
    """H.264 + yuv420p via ffmpeg stdin; plays reliably in VS Code / browsers."""

    def __init__(self, path, fps, size, out_pix_fmt="yuv420p", crf="23", preset="medium"):
        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid frame size: {size}")

        cmd = [
            shutil.which("ffmpeg") or "ffmpeg",
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
            str(out_pix_fmt),
            "-crf",
            str(crf),
            "-preset",
            str(preset),
            path,
        ]
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


def _open_visualization_writer(path, fps, size):
    if shutil.which("ffmpeg"):
        return _FfmpegH264VisualizationWriter(path, fps, size), "libx264/ffmpeg"
    writer, codec = _open_writer(path, fps, size)
    print("warning: ffmpeg not found; depth_visualization.mp4 may not preview in VS Code (try installing ffmpeg).")
    return writer, codec


def depth_to_vis(depth_m):
    valid = depth_m > 0
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    lo = np.percentile(depth_m[valid], 1.0)
    hi = np.percentile(depth_m[valid], 99.0)
    if hi <= lo:
        vis = np.zeros(depth_m.shape, dtype=np.uint8)
    else:
        norm = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
        vis = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


class _DepthWriter:
    """Write uint16 depth in a format compatible with BEHAVE/CARI4D tools."""

    def __init__(self, path, fps, size):
        if Uint16Writer is None:
            raise RuntimeError(
                "videoio.Uint16Writer is required to write BEHAVE-compatible depth format, "
                f"but videoio import failed: {_VIDEOIO_IMPORT_ERROR}"
            )
        try:
            self._uint16_writer = Uint16Writer(path, size, fps=int(round(fps)))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize videoio.Uint16Writer for BEHAVE-compatible depth output: {exc}"
            ) from exc

    @property
    def backend(self):
        return "videoio.Uint16Writer"

    def write(self, depth_u16):
        self._uint16_writer.write(depth_u16)

    def release(self):
        self._uint16_writer.close()


def _intrinsics_to_camera(intrinsics):
    if intrinsics is None:
        return None
    k = np.array(
        [
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return Pinhole(K=torch.from_numpy(k))


def _load_intrinsics_from_human_params(exp_dir):
    npz_path = osp.join(exp_dir, "human", "human_params.npz")
    if not osp.isfile(npz_path):
        raise FileNotFoundError(f"Missing intrinsics file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "intrinsics" not in data:
        raise KeyError(f"'intrinsics' not found in: {npz_path}")

    intr = np.asarray(data["intrinsics"]).reshape(-1)
    if intr.size < 4:
        raise ValueError(f"Invalid intrinsics in {npz_path}: expected >=4 values, got {intr.size}")

    fx, fy, cx, cy = [float(v) for v in intr[:4]]
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def _resolve_io_paths(args):
    if args.exp_dir:
        exp_dir = osp.abspath(args.exp_dir)
        input_mp4 = osp.join(exp_dir, "video.mp4")
        if not osp.isfile(input_mp4):
            raise FileNotFoundError(f"Input video not found at exp_dir/video.mp4: {input_mp4}")
        output_dir = osp.join(exp_dir, "processed")
        intrinsics = _load_intrinsics_from_human_params(exp_dir)
        return input_mp4, output_dir, intrinsics

    if not args.input_mp4 or not args.output_dir:
        raise ValueError("Use --exp_dir or provide both --input_mp4 and --output_dir.")

    intrinsics = None
    if None not in (args.fx, args.fy, args.cx, args.cy):
        intrinsics = np.array([args.fx, args.fy, args.cx, args.cy], dtype=np.float32)
    return args.input_mp4, args.output_dir, intrinsics


def run_inference(args):
    input_mp4, output_dir, intrinsics = _resolve_io_paths(args)
    if not osp.isfile(input_mp4):
        raise FileNotFoundError(f"Input mp4 not found: {input_mp4}")

    os.makedirs(output_dir, exist_ok=True)
    depth_path = osp.join(output_dir, "depth.mp4")
    if args.exp_dir:
        depth_vis_path = osp.join(osp.abspath(args.exp_dir), "depth_visualization.mp4")
    else:
        depth_vis_path = osp.join(output_dir, "depth_visualization.mp4")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{args.model_name}").to(device).eval()
    setattr(model, "resolution_level", args.resolution_level)

    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else float(args.fallback_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    depth_writer = _DepthWriter(depth_path, fps, size)
    vis_writer, vis_codec = _open_visualization_writer(depth_vis_path, fps, size)
    print(f"depth.mp4 writer: {depth_writer.backend}, depth visualization: {vis_codec}")
    print(f"processing: {input_mp4} ({width}x{height}, {fps:.3f} fps, {total} frames)")
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics.tolist()
        print(f"intrinsics: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")

    pbar = tqdm(total=total if total > 0 else None, desc="UniDepth", unit="frame")
    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(device)

            # UniDepth may update camera internals during infer; rebuild per-frame
            # to avoid accumulating numeric drift/singular K.
            camera = _intrinsics_to_camera(intrinsics)
            pred = model.infer(rgb, camera)
            depth_m = pred["depth"][0, 0].detach().cpu().numpy()
            depth_u16 = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)

            depth_writer.write(depth_u16)
            vis_writer.write(depth_to_vis(depth_m))
            pbar.update(1)

    pbar.close()
    cap.release()
    depth_writer.release()
    vis_writer.release()
    print(f"saved: {depth_path}")
    print(f"saved: {depth_vis_path}")
    print("note: depth.mp4 stores depth in uint16 millimeters using videoio.Uint16Writer (depth-reg-compatible format).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UniDepth on an input mp4 and save depth.mp4 + depth_visualization.mp4."
    )
    parser.add_argument("--exp_dir", type=str, default="", help="Experiment dir containing video.mp4 and human/human_params.npz.")
    parser.add_argument("--input_mp4", type=str, default="", help="Input RGB video path.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory.")
    parser.add_argument("--model_name", type=str, default="unidepth-v2-vitl14")
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fallback_fps", type=float, default=30.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
