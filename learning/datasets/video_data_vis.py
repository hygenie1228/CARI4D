from __future__ import annotations

import os
import os.path as osp
import re
import json
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import joblib
import kornia
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import Utils
from behave_data.utils import init_video_controllers
from tools import img_utils
from Utils import load_smpl_obj_uvmap
from lib_smpl import SMPL_ASSETS_ROOT, get_smpl
from lib_smpl.body_landmark import BodyLandmarks
from behave_data.const import _sub_gender


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _patch_utils_nvdiffrast_render_ob_in_cams_device() -> None:
    """Ensure ``ob_in_cams`` is CUDA before calling Utils.nvdiffrast_render.

    This keeps ``run_horefine`` unchanged while avoiding CPU/CUDA matmul mismatches in
    ``Utils.nvdiffrast_render`` when callers pass ``ob_in_cams`` as CPU tensors.
    """
    if getattr(Utils.nvdiffrast_render, "_horefine_obincams_patched", False):
        return
    _orig = Utils.nvdiffrast_render

    def _wrapped(*args: Any, **kwargs: Any):
        if torch.cuda.is_available():
            if "ob_in_cams" in kwargs:
                kwargs["ob_in_cams"] = torch.as_tensor(kwargs["ob_in_cams"], device="cuda", dtype=torch.float)
            elif len(args) >= 4:
                args = list(args)
                args[3] = torch.as_tensor(args[3], device="cuda", dtype=torch.float)
                args = tuple(args)
        return _orig(*args, **kwargs)

    _wrapped._horefine_obincams_patched = True
    Utils.nvdiffrast_render = _wrapped


_patch_utils_nvdiffrast_render_ob_in_cams_device()


def _horefine_camera_params_from_args(args: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bool(getattr(args, "wild_video", False)):
        data_source = str(getattr(args, "data_source", "behave"))
        if data_source == "behave":
            fx, fy = 979.7844, 979.840
            cx, cy = 1018.952, 779.486
        elif data_source == "intercap":
            from behave_data.const import ICAP_CENTERs, ICAP_FOCALs

            fx, fy = ICAP_FOCALs[0, 0], ICAP_FOCALs[0, 1]
            cx, cy = ICAP_CENTERs[0, 0], ICAP_CENTERs[0, 1]
        elif data_source == "hodome":
            from behave_data.const import HODOME_VIEW_IDS, get_camera_K_hodome

            K = get_camera_K_hodome(osp.basename(args.video), HODOME_VIEW_IDS[1])
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
        elif data_source == "imhd":
            from behave_data.const import IMHD_VIEW_IDS, get_IMHD_camera_K

            K = get_IMHD_camera_K(osp.basename(args.video), IMHD_VIEW_IDS[0])
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
        elif data_source == "procigen":
            K = np.array(
                [
                    [979.784, 0, 1018.952],
                    [0, 979.840, 779.486],
                    [0, 0, 1],
                ]
            )
            K[:2] /= 2.0
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
        else:
            raise ValueError(f"Invalid data source: {data_source}")
    else:
        pkl_file = str(getattr(args, "video")).replace(".mp4", ".pkl")
        d = joblib.load(pkl_file)
        fx, fy = d["fx"], d["fy"]
        cx, cy = d["cx"], d["cy"]
    # Keep default numpy float dtype (float64) to match BehaveRenderer/run_horefine numerics exactly.
    focal = np.array([fx, fy])
    principal = np.array([cx, cy])
    K_full = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return focal, principal, K_full


def _derive_seq_and_video_prefix(partial: Any) -> tuple[str, str]:
    seq_name = getattr(partial, "seq_name", None)
    video_prefix = getattr(partial, "video_prefix", None)
    if seq_name is not None and video_prefix is not None:
        return str(seq_name), str(video_prefix)
    args = getattr(partial, "args", None)
    video_path = str(getattr(args, "video", "") or "")
    if not video_path:
        raise RuntimeError("horefine_vis: seq_name/video_prefix missing in ctx and args.video is empty")
    vp = osp.basename(video_path).split(".")[0]
    if vp == "video":
        exp_base = osp.basename(osp.dirname(video_path.rstrip("/")))
        m = re.match(r"^(.+)_(\d+)$", exp_base)
        if m:
            vp = m.group(1)
    sq = vp
    return sq, vp


def _build_nlf_from_human_init_npz(human_npz_path: str, fp_frames: list[str]) -> dict[str, Any]:
    z = np.load(human_npz_path, allow_pickle=True)
    required = ["global_orient", "body_pose", "lhand_pose", "rhand_pose", "betas", "trans"]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"missing keys in {human_npz_path}: {missing}")

    go = np.asarray(z["global_orient"], dtype=np.float32)
    bp = np.asarray(z["body_pose"], dtype=np.float32)
    lh = np.asarray(z["lhand_pose"], dtype=np.float32)
    rh = np.asarray(z["rhand_pose"], dtype=np.float32)
    betas = np.asarray(z["betas"], dtype=np.float32)
    trans = np.asarray(z["trans"], dtype=np.float32)
    poses = np.concatenate([go, bp, lh, rh], axis=1).astype(np.float32)  # (T, 156)

    # fp_frames are usually zero-padded frame indices (e.g. "000030").
    # If parsing fails, fall back to leading contiguous frames.
    parsed: list[int] = []
    parse_ok = True
    for fr in fp_frames:
        s = osp.splitext(osp.basename(str(fr)))[0]
        try:
            parsed.append(int(s[1:]) if s.startswith("t") else int(s))
        except ValueError:
            parse_ok = False
            break

    if parse_ok:
        if len(parsed) == 0:
            raise ValueError("fp_frames is empty while building nlf_data")
        if max(parsed) >= len(poses):
            raise ValueError(
                f"human init npz has fewer frames ({len(poses)}) than requested max frame index ({max(parsed)})"
            )
        poses_sel = poses[parsed]
        betas_sel = betas[parsed]
        trans_sel = trans[parsed]
    else:
        n = len(fp_frames)
        if len(poses) < n:
            raise ValueError(f"human init npz has fewer frames ({len(poses)}) than fp_frames ({n})")
        poses_sel = poses[:n]
        betas_sel = betas[:n]
        trans_sel = trans[:n]

    gender = str(z["gender"].item()) if "gender" in z else "neutral"
    return {
        "poses": poses_sel[:, None, :].astype(np.float32),
        "betas": betas_sel[:, None, :].astype(np.float32),
        "transls": trans_sel[:, None, :].astype(np.float32),
        "frames": [str(x) for x in fp_frames],
        "gender": gender,
        "kids": [0],
    }


def _build_fp_from_object_init_npz(object_npz_path: str, fp_frames: list[str]) -> tuple[np.ndarray, list[str]]:
    z = np.load(object_npz_path, allow_pickle=True)
    required = ["angle", "trans"]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"missing keys in {object_npz_path}: {missing}")
    angles = np.asarray(z["angle"], dtype=np.float32)
    trans = np.asarray(z["trans"], dtype=np.float32)
    if angles.shape != trans.shape or angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(f"invalid object init shapes angle={angles.shape}, trans={trans.shape}")

    parsed: list[int] = []
    parse_ok = True
    for fr in fp_frames:
        s = osp.splitext(osp.basename(str(fr)))[0]
        try:
            parsed.append(int(s[1:]) if s.startswith("t") else int(s))
        except ValueError:
            parse_ok = False
            break

    if parse_ok and len(parsed) > 0 and max(parsed) < len(angles):
        a_sel = angles[parsed]
        t_sel = trans[parsed]
        frames_out = [str(x) for x in fp_frames]
    else:
        n = len(fp_frames)
        if len(angles) < n:
            raise ValueError(f"object init npz has fewer frames ({len(angles)}) than required ({n})")
        a_sel = angles[:n]
        t_sel = trans[:n]
        frames_out = [str(x) for x in fp_frames]

    fp_poses = np.repeat(np.eye(4, dtype=np.float32)[None, None, :, :], len(a_sel), axis=0)
    fp_poses[:, 0, :3, :3] = R.from_rotvec(a_sel).as_matrix().astype(np.float32)
    fp_poses[:, 0, :3, 3] = t_sel.astype(np.float32)
    return fp_poses, frames_out


class _MP4MaskLoader:
    """Local copy of the sequential MP4 mask loader."""

    def __init__(self, human_mask_mp4: str, object_mask_mp4: str, fps: float = 30.0):
        self.human_mask_mp4 = human_mask_mp4
        self.object_mask_mp4 = object_mask_mp4
        self.cap_h = cv2.VideoCapture(human_mask_mp4)
        self.cap_o = cv2.VideoCapture(object_mask_mp4)
        if not self.cap_h.isOpened() or not self.cap_o.isOpened():
            raise RuntimeError(
                f"failed to open mask videos: human={human_mask_mp4}, object={object_mask_mp4}"
            )
        self.fps = float(fps)
        self._idx = -1
        self._frame_h = None
        self._frame_o = None

    def _frame_index_from_time_str(self, frame_time: str) -> int:
        if isinstance(frame_time, str) and frame_time.startswith("t"):
            try:
                t = float(frame_time[1:])
                return int(round(t * self.fps))
            except ValueError:
                pass
        try:
            return int(frame_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported frame_time format: {frame_time}") from exc

    def _read_next(self) -> tuple[np.ndarray, np.ndarray]:
        ok_h, fh = self.cap_h.read()
        ok_o, fo = self.cap_o.read()
        if not ok_h or not ok_o or fh is None or fo is None:
            raise RuntimeError(f"mask videos ended early at frame index {self._idx + 1}")
        self._idx += 1
        self._frame_h = fh
        self._frame_o = fo
        return fh, fo

    def get_masks(self, frame_time: str) -> tuple[np.ndarray, np.ndarray]:
        target_idx = self._frame_index_from_time_str(frame_time)
        if target_idx < self._idx:
            raise RuntimeError(
                f"non-monotonic frame access for mp4 masks: target={target_idx}, current={self._idx}"
            )
        while self._idx < target_idx:
            self._read_next()
        if self._frame_h is None or self._frame_o is None:
            self._read_next()
        mask_h = (cv2.cvtColor(self._frame_h, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8) * 255
        mask_o = (cv2.cvtColor(self._frame_o, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8) * 255
        return mask_h, mask_o

    def fork(self) -> "_MP4MaskLoader":
        """Return a fresh sequential reader starting from frame 0."""
        return _MP4MaskLoader(self.human_mask_mp4, self.object_mask_mp4, fps=self.fps)


def _horefine_prepare_video_mask_loader(args: Any, kids: list, video_prefix: str, cfg: Any) -> tuple[list, Any]:
    args.nodepth = False
    controllers, _ = init_video_controllers(args, args.video, kids)
    masks_root = str(_cfg_get(cfg, "masks_root", ""))
    human_mask_mp4 = None
    object_mask_mp4 = None
    if "," in masks_root:
        left, right = masks_root.split(",", 1)
        if left.strip().lower().endswith(".mp4") and right.strip().lower().endswith(".mp4"):
            human_mask_mp4 = left.strip()
            object_mask_mp4 = right.strip()
    else:
        # MP4-only mode: prefer explicit files in masks_root directory, then near input video.
        if masks_root:
            hm = osp.join(masks_root, "human_mask.mp4")
            om = osp.join(masks_root, "object_mask.mp4")
            if osp.isfile(hm) and osp.isfile(om):
                human_mask_mp4, object_mask_mp4 = hm, om
        if human_mask_mp4 is None or object_mask_mp4 is None:
            base = osp.dirname(str(args.video))
            hm = osp.join(base, "human_mask.mp4")
            om = osp.join(base, "object_mask.mp4")
            if osp.isfile(hm) and osp.isfile(om):
                human_mask_mp4, object_mask_mp4 = hm, om
    if human_mask_mp4 is None or object_mask_mp4 is None:
        raise FileNotFoundError(
            "MP4 mask files are required. Set cfg.masks_root to "
            "'/path/human_mask.mp4,/path/object_mask.mp4' or provide files named "
            "'human_mask.mp4' and 'object_mask.mp4' in masks_root/ or video directory."
        )
    print(f"loading masks from mp4: {human_mask_mp4}, {object_mask_mp4}")
    tar_mask = _MP4MaskLoader(human_mask_mp4, object_mask_mp4, fps=float(getattr(args, "fps", 30)))
    return controllers, tar_mask


def _frame_cache_file(frame_cache_dir: str, idx: int) -> str:
    return osp.join(frame_cache_dir, f"{idx:06d}.npz")


def _frame_cache_meta_file(frame_cache_dir: str) -> str:
    return osp.join(frame_cache_dir, "_meta.json")


def _load_cached_rgbd_frame(frame_cache_dir: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(_frame_cache_file(frame_cache_dir, idx)) as d:
        return d["color"], d["depth"]


def _ensure_rgbd_frame_cache(
    *,
    frame_cache_dir: str,
    fp_frames: list,
    controllers: list,
    kids: list,
    enum_idx: int,
    fps: float = 30.0,
) -> None:
    cache_version = 1
    os.makedirs(frame_cache_dir, exist_ok=True)
    meta_fp = _frame_cache_meta_file(frame_cache_dir)
    rebuild_cache = True
    if osp.isfile(meta_fp):
        try:
            meta = json.load(open(meta_fp, "r"))
            rebuild_cache = not (
                int(meta.get("version", -1)) == cache_version
                and int(meta.get("num_frames", -1)) == int(len(fp_frames))
            )
            if not rebuild_cache and _cache_files_complete(len(fp_frames), lambda i: _frame_cache_file(frame_cache_dir, i)):
                print(f"[cache] rgbd hit: {frame_cache_dir}")
                return
        except Exception:
            rebuild_cache = True
    if rebuild_cache:
        for fn in os.listdir(frame_cache_dir):
            if fn.endswith(".npz") or fn == "_meta.json":
                try:
                    os.remove(osp.join(frame_cache_dir, fn))
                except OSError:
                    pass
    itr = tqdm(
        enumerate(fp_frames),
        total=len(fp_frames),
        desc=f"[cache] rgbd cam{enum_idx}",
        leave=True,
        dynamic_ncols=True,
    )
    for idx, frame_time in itr:
        cache_fp = _frame_cache_file(frame_cache_dir, idx)
        if osp.isfile(cache_fp):
            continue
        t = float(str(frame_time)[1:])
        actual_times = np.array([controllers[x].get_closest_time(t) for x, _ in enumerate(kids)])
        best_kid = np.argmin(np.abs(actual_times - t))
        actual_time = actual_times[best_kid]
        color, depth = controllers[enum_idx].get_closest_frame(actual_time)
        np.savez_compressed(cache_fp, color=np.asarray(color, dtype=np.uint8), depth=np.asarray(depth))
    with open(meta_fp, "w") as f:
        json.dump({"version": cache_version, "num_frames": int(len(fp_frames))}, f)


def _mask_cache_file(mask_cache_dir: str, idx: int) -> str:
    return osp.join(mask_cache_dir, f"{idx:06d}.npz")


def _mask_cache_meta_file(mask_cache_dir: str) -> str:
    return osp.join(mask_cache_dir, "_meta.json")


def _load_cached_masks(mask_cache_dir: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(_mask_cache_file(mask_cache_dir, idx)) as d:
        return d["mask_h"], d["mask_o"]


def _render_cache_file(render_cache_dir: str, idx: int) -> str:
    return osp.join(render_cache_dir, f"{idx:06d}.npz")


def _render_cache_meta_file(render_cache_dir: str) -> str:
    return osp.join(render_cache_dir, "_meta.json")


def _load_cached_render_inputs(render_cache_dir: str, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(_render_cache_file(render_cache_dir, idx)) as d:
        return d["rgb"], d["input_xyz"], d["render_rgb"], d["render_xyz"]


def _cache_files_complete(num_frames: int, file_fn) -> bool:
    return all(osp.isfile(file_fn(i)) for i in range(int(num_frames)))


def _ensure_mask_frame_cache(
    *,
    mask_cache_dir: str,
    fp_frames: list,
    mask_loader: _MP4MaskLoader,
) -> None:
    cache_version = 1
    os.makedirs(mask_cache_dir, exist_ok=True)
    meta_fp = _mask_cache_meta_file(mask_cache_dir)
    rebuild_cache = True
    if osp.isfile(meta_fp):
        try:
            meta = json.load(open(meta_fp, "r"))
            rebuild_cache = not (
                int(meta.get("version", -1)) == cache_version
                and int(meta.get("num_frames", -1)) == int(len(fp_frames))
            )
            if not rebuild_cache and _cache_files_complete(len(fp_frames), lambda i: _mask_cache_file(mask_cache_dir, i)):
                print(f"[cache] masks hit: {mask_cache_dir}")
                return
        except Exception:
            rebuild_cache = True
    if rebuild_cache:
        for fn in os.listdir(mask_cache_dir):
            if fn.endswith(".npz") or fn == "_meta.json":
                try:
                    os.remove(osp.join(mask_cache_dir, fn))
                except OSError:
                    pass
    itr = tqdm(
        enumerate(fp_frames),
        total=len(fp_frames),
        desc="[cache] masks",
        leave=True,
        dynamic_ncols=True,
    )
    for idx, frame_time in itr:
        cache_fp = _mask_cache_file(mask_cache_dir, idx)
        if osp.isfile(cache_fp):
            continue
        mask_h, mask_o = mask_loader.get_masks(str(frame_time))
        np.savez_compressed(
            cache_fp,
            mask_h=np.asarray(mask_h, dtype=np.uint8),
            mask_o=np.asarray(mask_o, dtype=np.uint8),
        )
    with open(meta_fp, "w") as f:
        json.dump({"version": cache_version, "num_frames": int(len(fp_frames))}, f)


def _horefine_get_smpl_diameter(betas_avg: np.ndarray, smpl_model: Any) -> float:
    verts_tpose = smpl_model(
        torch.zeros(1, 156).cuda(),
        torch.from_numpy(betas_avg[None]).cuda(),
        torch.from_numpy(np.zeros((1, 3))).cuda(),
    )[0].cpu().numpy()
    np.random.seed(0)
    samples = trimesh.Trimesh(verts_tpose[0], smpl_model.faces).sample(8000)
    return Utils.compute_mesh_diameter(model_pts=samples, n_sample=8000)


def _horefine_kroi_from_corners(
    bottom_right: np.ndarray,
    top_left: np.ndarray,
    render_size: tuple,
    focal: np.ndarray,
    principal_point: np.ndarray,
) -> np.ndarray:
    crop_size = np.mean(bottom_right - top_left)
    scale = float(render_size[0]) / float(crop_size)
    focal_roi = focal * scale
    principal_roi = (principal_point - top_left) * scale
    return np.array(
        [
            [focal_roi[0], 0, principal_roi[0]],
            [0, focal_roi[1], principal_roi[1]],
            [0, 0, 1.0],
        ]
    )


def _horefine_crop_color_dmap(
    bbox: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    render_size: tuple,
    K_full: np.ndarray,
) -> tuple:
    bmin, bmax = bbox[:2], bbox[2:]
    crop_size = np.max(bmax - bmin)
    crop_center = (bmin + bmax) / 2
    top_left = crop_center - crop_size / 2
    bottom_right = crop_center + crop_size / 2
    left = torch.tensor([top_left[0]])
    right = torch.tensor([bottom_right[0]])
    top = torch.tensor([top_left[1]])
    bottom = torch.tensor([bottom_right[1]])
    tf_full = Utils.compute_tf_batch(left=left, right=right, top=top, bottom=bottom, out_size=render_size).cpu()
    dmap_xyz = Utils.depth2xyzmap(depth / 1000.0, K_full)
    valid = depth > 0
    dmap_xyz[~valid] = 0
    dmap_xyz = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(dmap_xyz[None], device="cpu", dtype=torch.float).permute(0, 3, 1, 2),
        tf_full,
        dsize=render_size,
        mode="nearest",
        align_corners=False,
    )[0].permute(1, 2, 0)
    rgbm = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(color[None], device="cpu", dtype=torch.float).permute(0, 3, 1, 2),
        tf_full,
        dsize=render_size,
        mode="nearest",
        align_corners=False,
    )[0].permute(1, 2, 0)
    return dmap_xyz, rgbm


def _horefine_get_one_channel_mask(mask_ho: torch.Tensor) -> torch.Tensor:
    mask_h = mask_ho[0] > 0.5
    mask_o = mask_ho[1] > 0.5
    out = torch.zeros_like(mask_ho[0], dtype=torch.float)
    out[mask_h] = 1.0
    out[mask_o] = 2.0
    out[mask_h & mask_o] = 3.0
    return out


def _horefine_process_input(
    *,
    dmap_xyz_init: torch.Tensor,
    i: int,
    input_data: dict,
    mesh_diameter: float,
    nlf_transl: np.ndarray,
    pose_init: np.ndarray,
    render_data: dict,
    rgb_render: np.ndarray,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.from_numpy(input_data["rgbmB"][:, :, :3] / 255.0).permute(2, 0, 1).float()
    dmap_xyz = torch.from_numpy(input_data["xyzB"]).permute(2, 0, 1).float()
    if "behave-fp+input" in str(_cfg_get(cfg, "render_root", "")):
        dmap_xyz /= 1000.0
    if bool(_cfg_get(cfg, "normalize_xyz", False)):
        dmap_xyz *= 2 / mesh_diameter
        dmap_xyz_init *= 2 / mesh_diameter
    if bool(_cfg_get(cfg, "add_ho_mask", False)):
        mask_ho = torch.from_numpy(input_data["rgbmB"][:, :, 3:] / 255.0).permute(2, 0, 1).float()
        mask_encode_type = str(_cfg_get(cfg, "mask_encode_type", "stack"))
        if mask_encode_type in {"stack", "stack-occ"}:
            dmap_xyz = torch.cat([dmap_xyz, mask_ho], axis=0)
        elif mask_encode_type == "hum-obj-fullobj":
            mask_render_obj_full = render_data["mask_o"][:, :, 1][None]
            dmap_xyz = torch.cat([dmap_xyz, mask_ho, torch.from_numpy(mask_render_obj_full).float()], 0)
        elif mask_encode_type == "obj-fullobj":
            mask_render_obj_full = render_data["mask_o"][:, :, 1][None]
            dmap_xyz = torch.cat([dmap_xyz, mask_ho[1:], torch.from_numpy(mask_render_obj_full).float()], 0)
        elif mask_encode_type == "one-channel":
            dmap_xyz = torch.cat([dmap_xyz, _horefine_get_one_channel_mask(mask_ho)[None]], axis=0)
        else:
            raise ValueError("Unknown mask encode type: " + mask_encode_type)
    if bool(_cfg_get(cfg, "mask_rgb_bkg", False)):
        if not bool(_cfg_get(cfg, "add_ho_mask", False)):
            raise AssertionError("mask_rgb_bkg requires add_ho_mask=True")
        mask_fore = ((mask_ho[0:1] > 0.5) | (mask_ho[1:2] > 0.5)).expand(3, -1, -1)
        rgb[~mask_fore] = 0.0
        dmap_xyz[:3][~mask_fore] = 0.0
    if bool(_cfg_get(cfg, "add_ho_mask", False)):
        mask_render_full = np.mean(rgb_render, -1) > 0.01
        mask_render_obj = np.asarray(render_data["mask_o"], dtype=np.float32)
        if len(mask_render_obj.shape) == 3:
            mask_render_obj = mask_render_obj[:, :, 0]
        obj_fg = mask_render_obj > 0.5
        mask_render_hum = mask_render_full & (~obj_fg)
        mask_ho_render = torch.from_numpy(np.stack([mask_render_hum, mask_render_obj], 0)).float()
        mask_encode_type = str(_cfg_get(cfg, "mask_encode_type", "stack"))
        if mask_encode_type == "stack":
            dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render], 0)
        elif mask_encode_type == "one-channel":
            dmap_xyz_a = torch.cat([dmap_xyz_init, _horefine_get_one_channel_mask(mask_ho_render)[None]], axis=0)
        elif mask_encode_type == "hum-obj-fullobj":
            mask_render_obj_full = render_data["mask_o"][:, :, 1][None]
            dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render, torch.from_numpy(mask_render_obj_full).float()], 0)
        elif mask_encode_type == "obj-fullobj":
            mask_render_obj_full = render_data["mask_o"][:, :, 1][None]
            dmap_xyz_a = torch.cat(
                [dmap_xyz_init, mask_ho_render[1:], torch.from_numpy(mask_render_obj_full).float()],
                0,
            )
        elif mask_encode_type == "stack-occ":
            mask_o_render = torch.from_numpy(mask_render_obj)
            mask_o_render[mask_ho[0] > 0.5] = 0
            dmap_xyz_a = torch.cat([dmap_xyz_init, torch.stack([mask_ho[0], mask_o_render], 0)], 0)
        else:
            raise ValueError("Unknown mask encode type: " + mask_encode_type)
    else:
        dmap_xyz_a = dmap_xyz_init.clone()
    if bool(_cfg_get(cfg, "subtract_transl", False)):
        invalid = dmap_xyz_a[2:3] < 0.01
        trans_ref = torch.from_numpy(pose_init[:3, 3]).reshape((3, 1, 1)) * 2 / mesh_diameter
        nlf_root = _cfg_get(cfg, "nlf_root", None)
        if nlf_root is not None:
            tr = str(_cfg_get(cfg, "trans_ref_type", "frame"))
            if tr == "frame":
                trans_ref = torch.from_numpy(nlf_transl[i].copy()).reshape((3, 1, 1)) * 2 / mesh_diameter
            elif tr == "1st-frame":
                trans_ref = torch.from_numpy(nlf_transl[0].copy()).reshape((3, 1, 1)) * 2 / mesh_diameter
            else:
                raise ValueError(f"Unknown translation reference type: {tr}")
        dmap_xyz_a[:3] = dmap_xyz_a[:3] - trans_ref
        dmap_xyz_a[:3][invalid.repeat(3, 1, 1)] = 0.0
        invalid = dmap_xyz[2:3] < 0.01
        dmap_xyz[:3] = dmap_xyz[:3] - trans_ref
        dmap_xyz[:3][invalid.repeat(3, 1, 1)] = 0.0
        if bool(_cfg_get(cfg, "crop_xyz_3d", False)):
            bound_min, bound_max = np.array([-1, -1, -1.0]), np.array([1, 1, 1.0])
            m = (
                (dmap_xyz[0] < bound_max[0])
                & (dmap_xyz[0] > bound_min[0])
                & (dmap_xyz[1] < bound_max[1])
                & (dmap_xyz[1] > bound_min[1])
                & (dmap_xyz[1] < bound_max[2])
                & (dmap_xyz[1] > bound_min[2])
            )
            dmap_xyz[:3][~m[None].repeat(3, 1, 1)] = 0.0
    return dmap_xyz, dmap_xyz_a, rgb


def _horefine_verts_from_prep(ctx: Any, prep: dict) -> np.ndarray:
    """Recover (T, Nv, 3) verts from cached prep tensors."""
    body_model = ctx.body_model
    device = ctx.device
    poses = prep["nlf_poses"].detach().to(device).float()
    betas = prep["betas_gt"][0].detach().to(device).float()
    trans = prep["nlf_transl"][0].detach().to(device).float()
    return body_model(poses, betas, trans)[0].detach().cpu().numpy()


def _materialize_horefine_vis_ctx_from_partial(partial: Any) -> None:
    """Mirror ``HORefineRunner.run_1seq`` setup (fp, meshes, masks, NLF, packed, body, landmark, mp4 fork).

    Mutates ``partial`` in place (same object as the early ``horefine_vis_ctx`` in ``run_horefine``).
    """
    import nvdiffrast.torch as dr

    args = partial.args
    cfg = partial.cfg
    device = partial.device
    seq_name, video_prefix = _derive_seq_and_video_prefix(partial)
    partial.seq_name = seq_name
    partial.video_prefix = video_prefix
    focal, principal_point, K_full = _horefine_camera_params_from_args(args)
    partial.focal = focal
    partial.principal_point = principal_point
    partial.K_full = K_full

    fp_poses = None
    fp_frames = None
    kids = [cfg.cam_id]
    w2c_rots = [np.eye(3) for _ in kids]
    w2c_trans = [np.zeros(3) for _ in kids]
    mesh_tensors, meshes = load_smpl_obj_uvmap(seq_name, use_hy3d=True, meshes_root=cfg.hy3d_meshes_root)
    meshes_any: Any = meshes
    glctx = dr.RasterizeCudaContext()
    render_size = (args.rend_size, args.rend_size)
    obj_idx = 1
    verts_obj_base_t = meshes_any[obj_idx].verts_padded()[0].to(device).float()
    tex = meshes[obj_idx].textures
    uv = tex.verts_uvs_padded()[0]
    uv[:, 1] = 1 - uv[:, 1]
    mesh_tensors_obj = {
        "tex": tex.maps_padded().to(device).float(),
        "uv_idx": torch.tensor(tex.faces_uvs_padded()[0], device=device, dtype=torch.int),
        "uv": uv.to(device).float(),
        "pos": meshes[obj_idx].verts_padded()[0].to(device).float(),
        "faces": torch.tensor(meshes[obj_idx].faces_padded()[0], device=device, dtype=torch.int),
        "vnormals": meshes[obj_idx].verts_normals_padded()[0].to(device).float(),
    }
    controllers, tar_mask = _horefine_prepare_video_mask_loader(args, kids, video_prefix, cfg)
    gt_to_perturb_pose = np.eye(4)
    if not cfg.wild_video:
        center = np.zeros(3)
    else:
        center = np.mean(verts_obj_base_t.detach().cpu().numpy(), axis=0)
    print("Using center: ", center)
    gt_to_perturb_pose[:3, 3] = center
    verts_obj_base = verts_obj_base_t.detach().cpu().numpy() - center
    enum_idx, kid = 0, cfg.cam_id
    save_dir_abs = osp.abspath(str(getattr(cfg, "save_dir", "experiments")))
    marker = f"{os.sep}data{os.sep}finetune_ckpts"
    if marker in save_dir_abs:
        exp_root = save_dir_abs.split(marker)[0]
    else:
        exp_root = osp.join(save_dir_abs, str(getattr(cfg, "exp_name", "debug")))
    frame_cache_dir = osp.join(exp_root, "data", "rgbd_frame_cache", seq_name, f"cam{kid}")
    mask_cache_dir = osp.join(exp_root, "data", "mask_frame_cache", seq_name, f"cam{kid}")
    render_cache_dir = osp.join(exp_root, "data", "render_frame_cache", seq_name, f"cam{kid}")
    human_init_npz = osp.join(exp_root, "human", "human_params_init.npz")
    object_init_npz = osp.join(exp_root, "object", "object_params_init.npz")
    if not osp.isfile(human_init_npz):
        raise FileNotFoundError(f"missing required init npz: {human_init_npz}")
    if not osp.isfile(object_init_npz):
        raise FileNotFoundError(f"missing required init npz: {object_init_npz}")

    packed = None
    packed_root = getattr(cfg, "packed_root", None)
    if packed_root:
        packed_file = osp.join(packed_root, f"{seq_name}_GT-packed.pkl")
        if osp.isfile(packed_file):
            packed = joblib.load(packed_file)
            print(f"loaded GT packed labels from {packed_file}")
            frames_packed_src = [str(x) for x in packed.get("frames", [])]
            if len(frames_packed_src) > 0:
                fp_frames = frames_packed_src
    if fp_frames is None:
        z_obj = np.load(object_init_npz, allow_pickle=True)
        n_obj = int(np.asarray(z_obj["angle"]).shape[0])
        fp_frames = [f"{i:06d}" for i in range(n_obj)]

    fp_poses, fp_frames = _build_fp_from_object_init_npz(object_init_npz, fp_frames)
    fp_frame_to_idx = {str(f): i for i, f in enumerate(fp_frames)}
    nlf_data = _build_nlf_from_human_init_npz(human_init_npz, fp_frames)
    _ensure_rgbd_frame_cache(
        frame_cache_dir=frame_cache_dir,
        fp_frames=fp_frames,
        controllers=controllers,
        kids=kids,
        enum_idx=enum_idx,
        fps=float(getattr(args, "fps", 30)),
    )
    _ensure_mask_frame_cache(
        mask_cache_dir=mask_cache_dir,
        fp_frames=fp_frames,
        mask_loader=tar_mask.fork(),
    )
    if packed is None:
        packed = {}
        packed["obj_angles"] = R.from_matrix(fp_poses[:, enum_idx, :3, :3]).as_rotvec().astype(np.float32)
        packed["obj_trans"] = fp_poses[:, enum_idx, :3, 3].astype(np.float32)
        packed["poses"] = nlf_data["poses"][:, 0].copy()
        packed["betas"] = nlf_data["betas"][:, 0].copy()
        packed["trans"] = nlf_data["transls"][:, 0].copy()
        packed["frames"] = fp_frames

    frames_packed = packed["frames"]
    frames_packed = [x for x in frames_packed if x in nlf_data["frames"]]
    body_model = get_smpl(_sub_gender[video_prefix.split("_")[1]], hands=True).to(device)
    betas_avg = np.mean(packed["betas"].reshape((-1, 10)), 0)
    mesh_diameter = _horefine_get_smpl_diameter(betas_avg, body_model)

    out_root = cfg.video_out
    os.makedirs(out_root, exist_ok=True)
    landmark = BodyLandmarks(SMPL_ASSETS_ROOT)

    vis_input = False
    if vis_input:
        # Input overlay path matches ``run_1seq``; do not ``imageio.get_writer`` here — the main loop
        # already holds ``vw_input`` for that path. Set ``partial.vw_input`` yourself if you need vis-batch appends.
        vw_input = getattr(partial, "vw_input", None)
    else:
        vw_input = None

    tar_mask_independent = tar_mask
    _masks_root_vis = str(getattr(cfg, "masks_root", ""))
    if "," in _masks_root_vis:
        _mh, _mo = _masks_root_vis.split(",", 1)
        if _mh.strip().lower().endswith(".mp4") and _mo.strip().lower().endswith(".mp4"):
            tar_mask_independent = _MP4MaskLoader(
                _mh.strip(), _mo.strip(), fps=float(getattr(args, "fps", 30))
            )

    partial.kid = kid
    partial.enum_idx = enum_idx
    partial.fp_frame_to_idx = fp_frame_to_idx
    partial.frame_cache_dir = frame_cache_dir
    partial.mask_cache_dir = mask_cache_dir
    partial.render_cache_dir = render_cache_dir
    partial.kids = kids
    partial.packed = packed
    partial.nlf_data = nlf_data
    partial.fp_poses = fp_poses
    partial.fp_frames = fp_frames
    partial.frames_packed = frames_packed
    partial.mesh_tensors = mesh_tensors
    partial.mesh_tensors_obj = mesh_tensors_obj
    partial.glctx = glctx
    partial.render_size = render_size
    partial.verts_obj_base = verts_obj_base
    partial.gt_to_perturb_pose = gt_to_perturb_pose
    partial.controllers = controllers
    partial.tar_mask = tar_mask
    partial.tar_mask_independent = tar_mask_independent
    partial.body_model = body_model
    partial.landmark = landmark
    partial.mesh_diameter = mesh_diameter
    partial.w2c_rots = w2c_rots
    partial.w2c_trans = w2c_trans
    partial.vis_input = vis_input
    partial.vw_input = vw_input
    if not hasattr(partial, "record_independent_vis"):
        partial.record_independent_vis = False

    # Precompute expensive nvdiffrast-based render inputs once at loader materialization.
    os.makedirs(render_cache_dir, exist_ok=True)
    render_meta_fp = _render_cache_meta_file(render_cache_dir)
    render_cache_version = 1
    need_render_rebuild = True
    if osp.isfile(render_meta_fp):
        try:
            meta = json.load(open(render_meta_fp, "r"))
            need_render_rebuild = not (
                int(meta.get("version", -1)) == render_cache_version
                and int(meta.get("num_frames", -1)) == int(len(frames_packed))
            )
            if not need_render_rebuild and _cache_files_complete(
                len(frames_packed), lambda i: _render_cache_file(render_cache_dir, i)
            ):
                print(f"[cache] render hit: {render_cache_dir}")
                need_render_rebuild = False
            elif not need_render_rebuild:
                need_render_rebuild = True
        except Exception:
            need_render_rebuild = True
    if need_render_rebuild:
        for fn in os.listdir(render_cache_dir):
            if fn.endswith(".npz") or fn == "_meta.json":
                try:
                    os.remove(osp.join(render_cache_dir, fn))
                except OSError:
                    pass
        clip_len = int(getattr(cfg, "clip_len", 16))
        for start in tqdm(
            range(0, len(frames_packed), clip_len),
            desc="[cache] render",
            leave=True,
            dynamic_ncols=True,
        ):
            end = min(start + clip_len, len(frames_packed))
            _ = horefine_vis_rebuild_independent_window_batch(partial, start, end)
        with open(render_meta_fp, "w") as f:
            json.dump({"version": render_cache_version, "num_frames": int(len(frames_packed))}, f)


def hydrate_horefine_vis_ctx(partial: Any) -> Any:
    """If ``partial`` has no ``packed`` yet, run the same init as ``run_horefine.run_1seq`` (fp→masks→NLF…)."""
    if getattr(partial, "packed", None) is not None:
        return partial
    _materialize_horefine_vis_ctx_from_partial(partial)
    return partial


def finalize_horefine_vis_ctx(ctx: Any) -> None:
    """Register ``ctx`` in ``ctx._vis_ctx_ref`` so :class:`HoRefineVisBatchLoader` can bind lazily.

    Use when the loader is constructed with ``ctx_ref=`` **before** ``ctx`` exists, or whenever
    you replace the namespace: call after ``horefine_vis_ctx = SimpleNamespace(..., _vis_ctx_ref=ref)``.
    """
    ref = getattr(ctx, "_vis_ctx_ref", None)
    if ref is not None:
        ref["ctx"] = ctx


def horefine_vis_window_build_prep_and_smpl_init(
    *,
    device: str,
    packed: dict,
    nlf_data: dict,
    frames_packed: list,
    start: int,
    end: int,
    enum_idx: int,
    body_model: Any,
    landmark: Any,
) -> dict[str, Any]:
    nlf_inds = np.array([nlf_data["frames"].index(x.split("/")[-1]) for x in frames_packed[start:end]])
    poses_nlf_init = nlf_data["poses"][nlf_inds, enum_idx].astype(np.float32)
    trans_nlf_init = nlf_data["transls"][nlf_inds, enum_idx].astype(np.float32)
    betas_nlf_init = nlf_data["betas"][:, enum_idx].copy()
    betas_avg = np.mean(betas_nlf_init, axis=0)[None].repeat(len(poses_nlf_init), axis=0)
    poses_full = packed["poses"][start:end].astype(np.float32)
    betas_gt = packed["betas"][start:end].astype(np.float32)
    verts_nlf_render_init = body_model(
        torch.from_numpy(poses_nlf_init).to(device),
        torch.from_numpy(betas_gt).to(device),
        torch.from_numpy(trans_nlf_init).to(device),
    )[0].cpu().numpy()
    joints_nlf_np = landmark.get_body_kpts_batch(verts_nlf_render_init)
    NJ = 52
    nlf_rot_np_init = (
        R.from_rotvec(poses_nlf_init.reshape(-1, 3)).as_matrix().astype(np.float32).reshape(-1, NJ, 3, 3)
    )
    prep = {
        "joints_nlf": torch.from_numpy(joints_nlf_np)[None].to(device).float(),
        "nlf_rotmat": torch.from_numpy(nlf_rot_np_init).to(device).float()[:, :24],
        "nlf_transl": torch.from_numpy(trans_nlf_init)[None].to(device).float(),
        "betas_gt": torch.from_numpy(betas_gt)[None].to(device).float(),
        "betas_nlf": torch.from_numpy(betas_avg)[None].to(device).float(),
        "nlf_poses": torch.from_numpy(poses_nlf_init).to(device).float(),
    }
    return {
        "prep": prep,
        "poses_full": poses_full,
        "betas_gt": betas_gt,
        "verts_nlf_render_init": verts_nlf_render_init,
        "trans_nlf_init": trans_nlf_init,
        "poses_nlf_init": poses_nlf_init,
        "betas_avg": betas_avg,
        # Same as run_horefine.run_1seq local NLF indexing (for batch-driven reload).
        "nlf_inds": nlf_inds,
        "betas_nlf_init": betas_nlf_init,
    }


def horefine_vis_window_load_rgb_masks_depth(
    *,
    start: int,
    end: int,
    frames_packed: list,
    fp_frames: list,
    seq_name: str,
    fp_poses: np.ndarray,
    enum_idx: int,
    gt_to_perturb_pose: np.ndarray,
    tar_mask_for_samples: Any,
    video_prefix: str,
    kid: int,
    controllers: Optional[list],
    kids: list,
    render_size: tuple,
    focal: np.ndarray,
    principal_point: np.ndarray,
    K_full: np.ndarray,
    fp_frame_to_idx: Optional[dict[str, int]] = None,
    frame_cache_dir: Optional[str] = None,
    mask_cache_dir: Optional[str] = None,
    tqdm_frames: bool = False,
) -> tuple[list, list, list, list, list, list, list]:
    input_rgbms, input_xyzs, bboxes = [], [], []
    poses_perturbed: list = []
    K_rois: list = []
    frames_used: list = []
    full_colors: list = []
    fp_indices: list = []
    itr = tqdm(range(start, end)) if tqdm_frames else range(start, end)
    for i in itr:
        frame_time = frames_packed[i]
        if fp_frame_to_idx is not None:
            idx_fp = fp_frame_to_idx.get(str(frame_time), -1)
        elif frame_time in fp_frames:
            idx_fp = fp_frames.index(frame_time)
        else:
            idx_fp = -1
        if idx_fp < 0:
            print(f"Frame {frame_time} not found in FP frames!")
            continue
        frames_used.append(f"{seq_name}/{frame_time}")
        fp_indices.append(int(idx_fp))
        pose_fp = np.matmul(fp_poses[idx_fp, enum_idx], gt_to_perturb_pose)
        if mask_cache_dir is None or not osp.isdir(mask_cache_dir):
            raise FileNotFoundError(
                f"Mask frame cache directory missing for getitem: {mask_cache_dir}"
            )
        mask_cache_fp = _mask_cache_file(mask_cache_dir, idx_fp)
        if not osp.isfile(mask_cache_fp):
            raise FileNotFoundError(f"Mask frame cache file missing for getitem: {mask_cache_fp}")
        mask_h, mask_o = _load_cached_masks(mask_cache_dir, idx_fp)
        if mask_h is None:
            continue
        bmin, bmax = img_utils.masks2bbox([mask_h, mask_o])
        center_2d = (bmax + bmin) / 2
        radius = np.max(bmax - bmin) * 1.1 / 2
        top_left = center_2d - radius
        bottom_right = center_2d + radius
        K_roi = _horefine_kroi_from_corners(bottom_right, top_left, render_size, focal, principal_point)
        if frame_cache_dir is None or not osp.isdir(frame_cache_dir):
            raise FileNotFoundError(
                f"RGBD frame cache directory missing for getitem: {frame_cache_dir}"
            )
        cache_fp = _frame_cache_file(frame_cache_dir, idx_fp)
        if not osp.isfile(cache_fp):
            raise FileNotFoundError(f"RGBD frame cache file missing for getitem: {cache_fp}")
        color_np, depth = _load_cached_rgbd_frame(frame_cache_dir, idx_fp)
        full_colors.append(np.asarray(color_np))
        mask_h_np = mask_h.astype(np.uint8)
        mask_o_np = mask_o.astype(np.uint8)
        color = np.concatenate([color_np, mask_h_np[:, :, None], mask_o_np[:, :, None]], axis=-1)
        bbox = np.hstack((top_left.astype(np.float32), bottom_right.astype(np.float32)))
        dmap_xyz, rgbm = _horefine_crop_color_dmap(bbox, color, depth, render_size, K_full)
        input_rgbms.append(rgbm)
        input_xyzs.append(dmap_xyz)
        K_rois.append(K_roi)
        poses_perturbed.append(pose_fp.copy())
        bboxes.append(bbox)
    return input_rgbms, input_xyzs, bboxes, K_rois, poses_perturbed, frames_used, full_colors, fp_indices


def horefine_vis_window_render_and_make_batch(
    *,
    B_in_cams: np.ndarray,
    verts_nlf_render: np.ndarray,
    prep: dict,
    trans_nlf: np.ndarray,
    poses_perturbed: list,
    frames_used: list,
    full_colors: list,
    input_rgbms: list,
    input_xyzs: list,
    bboxes: list,
    K_rois: list,
    poses_full: np.ndarray,
    betas_gt: np.ndarray,
    packed: dict,
    start: int,
    end: int,
    mesh_tensors: dict,
    mesh_tensors_obj: dict,
    glctx: Any,
    render_size: tuple,
    verts_obj_base: np.ndarray,
    device: str,
    args: Any,
    cfg: Any,
    mesh_diameter: float,
    w2c_rots: list,
    w2c_trans: list,
    enum_idx: int,
    seq_name: str,
    kid: int,
    vis_input: bool,
    vw_input: Any,
    append_vis_input: bool,
    nlf_inds: np.ndarray,
    poses_nlf_init: np.ndarray,
    trans_nlf_init: np.ndarray,
    betas_nlf_init: np.ndarray,
    fp_indices: list[int],
    render_cache_dir: Optional[str],
) -> dict:
    """``betas_nlf_init`` is the raw NLF ``betas[:, cam]`` column; averaged+repeated for the batch key."""
    verts_obj_batch = [
        np.matmul(verts_obj_base, pose_fp[:3, :3].T) + pose_fp[:3, 3] for pose_fp in B_in_cams
    ]
    verts_hum_batch = verts_nlf_render.copy()
    input_rgbs_final, input_xyz_final = [], []
    render_rgbs, render_xyz = [], []
    if render_cache_dir is not None:
        os.makedirs(render_cache_dir, exist_ok=True)
    for fi in range(len(verts_obj_batch)):
        idx_fp = int(fp_indices[fi])
        cache_hit = False
        if render_cache_dir is not None:
            render_fp = _render_cache_file(render_cache_dir, idx_fp)
            if osp.isfile(render_fp):
                rgb_np, input_xyz_np, render_rgb_np, render_xyz_np = _load_cached_render_inputs(
                    render_cache_dir, idx_fp
                )
                rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).float()
                dmap_xyz = torch.from_numpy(np.ascontiguousarray(input_xyz_np)).float()
                dmap_xyz_a = torch.from_numpy(np.ascontiguousarray(render_xyz_np)).float()
                render_rgbs.append(np.ascontiguousarray(render_rgb_np).astype(np.float32))
                render_xyz.append(dmap_xyz_a)
                input_xyz_final.append(dmap_xyz)
                input_rgbs_final.append(rgb)
                cache_hit = True
        if cache_hit:
            continue
        vh = verts_hum_batch[fi]
        vo = verts_obj_batch[fi]
        mesh_tensors["pos"] = torch.from_numpy(np.concatenate([vh, vo], 0)).float().cuda()
        mesh_tensors_obj["pos"] = torch.from_numpy(vo).float().cuda()
        bbox2d_ori = torch.tensor([[0, 0.0, render_size[0], render_size[1]]], device=device).repeat(1, 1)
        ob_in_cam_eye = torch.as_tensor(np.eye(4)[None], device=device, dtype=torch.float)
        extra: dict = {}
        rgb_r, depth_r, _ = Utils.nvdiffrast_render(
            K=np.stack([K_rois[fi]], 0),
            H=render_size[1],
            W=render_size[0],
            ob_in_cams=ob_in_cam_eye,
            context="cuda",
            get_normal=False,
            glctx=glctx,
            mesh_tensors=mesh_tensors,
            output_size=render_size,
            bbox2d=bbox2d_ori,
            use_light=True,
            extra=extra,
        )
        rgb_obj, depth_obj, _ = Utils.nvdiffrast_render(
            K=np.stack([K_rois[fi]], 0),
            H=render_size[1],
            W=render_size[0],
            ob_in_cams=ob_in_cam_eye,
            context="cuda",
            get_normal=False,
            glctx=glctx,
            mesh_tensors=mesh_tensors_obj,
            output_size=render_size,
            bbox2d=bbox2d_ori,
            use_light=True,
            extra=extra,
        )
        rgbs = (rgb_r.cpu().numpy() * 255).astype(np.uint8)
        dmaps = depth_r.cpu().numpy()
        dmap_full = depth_r[0].cpu().numpy()
        dmap_obj = depth_obj[0].cpu().numpy()
        mask_rend_o = (dmap_obj <= dmap_full) & (dmap_obj > 0)
        mask_o_full = dmap_obj > 0
        dmap_xyz_init_np = Utils.depth2xyzmap(dmap_full, K_rois[fi])
        dmap_xyz_init = torch.from_numpy(dmap_xyz_init_np).permute(2, 0, 1).float()
        input_data = {
            "rgbmB": input_rgbms[fi].cpu().numpy().astype(np.uint8).copy(),
            "xyzB": input_xyzs[fi].cpu().numpy().astype(np.float16).copy(),
        }
        render_data = {
            "rgba": rgbs[0].copy(),
            "depth": dmaps[0].astype(np.float16).copy(),
            "K_roi": K_rois[fi],
            "bbox": bboxes[fi],
            "mask_o": np.stack([mask_rend_o, mask_o_full], -1).copy(),
        }
        rgb_render = render_data["rgba"]
        if vis_input and append_vis_input and vw_input is not None:
            vis = np.concatenate(
                [
                    input_data["rgbmB"][:, :, :3],
                    render_data["rgba"][:, :, :3],
                    cv2.addWeighted(
                        input_data["rgbmB"][:, :, :3],
                        0.5,
                        render_data["rgba"][:, :, :3],
                        0.5,
                        0,
                    ),
                ],
                1,
            )
            vw_input.append_data(vis)
        dmap_xyz, dmap_xyz_a, rgb = _horefine_process_input(
            dmap_xyz_init=dmap_xyz_init.clone(),
            i=fi,
            input_data=input_data,
            mesh_diameter=mesh_diameter,
            nlf_transl=trans_nlf,
            pose_init=poses_perturbed[fi],
            render_data=render_data,
            rgb_render=render_data["rgba"],
            cfg=cfg,
        )
        render_rgbs.append(rgb_render.copy().transpose(2, 0, 1) / 255.0)
        render_xyz.append(dmap_xyz_a)
        input_xyz_final.append(dmap_xyz)
        input_rgbs_final.append(rgb)
        if render_cache_dir is not None:
            np.savez_compressed(
                _render_cache_file(render_cache_dir, idx_fp),
                rgb=rgb.detach().cpu().numpy().astype(np.float32),
                input_xyz=dmap_xyz.detach().cpu().numpy().astype(np.float32),
                render_rgb=(rgb_render.copy().transpose(2, 0, 1) / 255.0).astype(np.float32),
                render_xyz=dmap_xyz_a.detach().cpu().numpy().astype(np.float32),
            )

    poseA_norm = B_in_cams.copy()
    poseA_norm[:, :3, 3] *= 2 / mesh_diameter
    mesh_diam_tensor = torch.as_tensor(
        [mesh_diameter] * len(poses_perturbed), device=device, dtype=torch.float
    )[None]
    trans_norm = torch.as_tensor(np.array(args.trans_normalizer), device=device, dtype=torch.float).repeat(
        len(poses_perturbed), 1
    )[None]
    if full_colors:
        full_hw = tuple(int(x) for x in full_colors[0].shape[:2])
    else:
        # Fallback when full-size frame is unavailable.
        full_hw = tuple(int(x) for x in input_rgbms[0].shape[:2])
    # Same as run_horefine: per-frame SMPL uses one averaged NLF shape repeated T times.
    betas_nlf_init = np.mean(betas_nlf_init, axis=0)[None].repeat(len(poses_nlf_init), axis=0)

    angles_gt = packed["obj_angles"][start:end].astype(np.float32)
    transl_gt = packed["obj_trans"][start:end].astype(np.float32)
    R_wc = torch.from_numpy(w2c_rots[enum_idx]).to(device).float()
    t_wc = torch.from_numpy(w2c_trans[enum_idx]).to(device).float()
    R_obj = torch.from_numpy(R.from_rotvec(angles_gt).as_matrix()).to(device).float()
    t_world = torch.from_numpy(transl_gt).to(device).float()
    R_cam = torch.matmul(R_wc[None].expand(end - start, -1, -1), R_obj)
    t_cam = torch.matmul(t_world, R_wc.T) + t_wc
    pose_gt_mat = torch.eye(4, device=device, dtype=torch.float)[None].repeat(end - start, 1, 1)
    pose_gt_mat[:, :3, :3] = R_cam
    pose_gt_mat[:, :3, 3] = t_cam
    pose_gt_batched = pose_gt_mat[None].clone()
    pose_perturbed_tensor = torch.from_numpy(B_in_cams)[None].float().cuda()
    delta_transl = pose_gt_batched[:, :, :3, 3] - pose_perturbed_tensor[:, :, :3, 3]
    delta_rot = torch.matmul(
        pose_gt_batched[:, :, :3, :3],
        pose_perturbed_tensor[:, :, :3, :3].permute(0, 1, 3, 2),
    )

    hum_pose_init = torch.from_numpy(np.ascontiguousarray(poses_nlf_init.astype(np.float32, copy=False))).float().cuda()[None]
    hum_transl_init = torch.from_numpy(np.ascontiguousarray(trans_nlf_init.astype(np.float32, copy=False))).float().cuda()[None]
    hum_betas_init = torch.from_numpy(np.ascontiguousarray(betas_nlf_init.astype(np.float32, copy=False))).float().cuda()[None]

    hum_pose_gt = torch.from_numpy(np.ascontiguousarray(poses_full.astype(np.float32, copy=False))).float().cuda()[None]
    hum_betas_gt = torch.from_numpy(np.ascontiguousarray(betas_nlf_init.astype(np.float32, copy=False))).float().cuda()[None]
    hum_transl_gt = torch.from_numpy(np.ascontiguousarray(trans_nlf_init.astype(np.float32, copy=False))).float().cuda()[None]

    batch = {
        ## Input
        "input_rgbs": torch.stack(input_rgbs_final, 0).cuda().float()[None],
        "render_rgbs": torch.from_numpy(np.stack(render_rgbs, axis=0)).cuda().float()[None],
        "input_xyz": torch.stack(input_xyz_final, 0).float().cuda()[None],
        "render_xyz": torch.stack(render_xyz, 0).float().cuda()[None],
        
        ## Init
        "hum_pose_init": hum_pose_init,
        "hum_transl_init": hum_transl_init,
        "hum_betas_init": hum_betas_init,
        "hum_joints_init": prep["joints_nlf"],
        "obj_pose_init": pose_perturbed_tensor,
        
        ## Target
        "hum_pose_gt": hum_pose_gt,
        "hum_betas_gt": hum_betas_gt,
        "hum_transl_gt": hum_transl_gt,
        "obj_pose_gt": pose_gt_batched,

        "delta_rot": delta_rot,
        "delta_transl": delta_transl,
        
        ## Meta
        "mesh_diameter": mesh_diam_tensor,
        "trans_normalizer": trans_norm.reshape(1, len(poses_perturbed), 3),
        "K_rois": torch.from_numpy(np.stack(K_rois)).float().cuda()[None],
        "bboxes": np.stack(bboxes).astype(np.float32),
        "full_hw": full_hw,
        "frames_used": list(frames_used),
        "full_colors": [np.asarray(x).copy() for x in full_colors],
    }
    
    return batch


def horefine_vis_rebuild_independent_window_batch(
    ctx: Any,
    start: int,
    end: int,
    *,
    B_in_cams_override: Optional[np.ndarray] = None,
    prep_override: Optional[dict] = None,
    verts_hum_override: Optional[np.ndarray] = None,
) -> dict:
    """Rebuild one window dict like ``run_1seq`` (fork mp4 mask reader when available)."""
    device = ctx.device
    cfg = ctx.cfg
    args = ctx.args
    seq_name = ctx.seq_name
    video_prefix = ctx.video_prefix
    kid = ctx.kid
    enum_idx = ctx.enum_idx
    kids = ctx.kids
    packed = ctx.packed
    nlf_data = ctx.nlf_data
    fp_poses = ctx.fp_poses
    fp_frames = ctx.fp_frames
    frames_packed = ctx.frames_packed
    mesh_tensors = ctx.mesh_tensors
    mesh_tensors_obj = ctx.mesh_tensors_obj
    glctx = ctx.glctx
    render_size = ctx.render_size
    verts_obj_base = ctx.verts_obj_base
    gt_to_perturb_pose = ctx.gt_to_perturb_pose
    tar_mask = ctx.tar_mask
    body_model = ctx.body_model
    landmark = ctx.landmark
    mesh_diameter = ctx.mesh_diameter
    w2c_rots = ctx.w2c_rots
    w2c_trans = ctx.w2c_trans
    focal = ctx.focal
    principal_point = ctx.principal_point
    K_full = ctx.K_full
    vis_input = ctx.vis_input
    vw_input = ctx.vw_input
    record_indie_vis = getattr(ctx, "record_independent_vis", False)
    fp_frame_to_idx = getattr(ctx, "fp_frame_to_idx", None)
    frame_cache_dir = getattr(ctx, "frame_cache_dir", None)
    mask_cache_dir = getattr(ctx, "mask_cache_dir", None)
    render_cache_dir = getattr(ctx, "render_cache_dir", None)

    # Prefer independent mask source. For MP4 readers, use a fresh instance per getitem
    # so repeated calls with the same [start, end] never hit backward frame access.
    mask_src = getattr(ctx, "tar_mask_independent", tar_mask)
    if hasattr(mask_src, "fork") and callable(getattr(mask_src, "fork")):
        mask_src = mask_src.fork()

    if prep_override is None:
        w = horefine_vis_window_build_prep_and_smpl_init(
            device=device,
            packed=packed,
            nlf_data=nlf_data,
            frames_packed=frames_packed,
            start=start,
            end=end,
            enum_idx=enum_idx,
            body_model=body_model,
            landmark=landmark,
        )
        prep = w["prep"]
        poses_full = w["poses_full"]
        betas_gt = w["betas_gt"]
        trans_nlf = w["trans_nlf_init"]
        verts_nlf_render = (
            verts_hum_override.copy() if verts_hum_override is not None else w["verts_nlf_render_init"]
        )
        nlf_inds = w["nlf_inds"]
        poses_nlf_init = w["poses_nlf_init"]
        trans_nlf_init = w["trans_nlf_init"]
        betas_nlf_init = w["betas_nlf_init"]
    else:
        prep = prep_override
        poses_full = packed["poses"][start:end].astype(np.float32)
        betas_gt = packed["betas"][start:end].astype(np.float32)
        if verts_hum_override is None:
            verts_nlf_render = _horefine_verts_from_prep(ctx, prep_override)
        else:
            verts_nlf_render = verts_hum_override.copy()
        trans_nlf = prep["nlf_transl"][0].detach().cpu().numpy()
        nlf_inds = np.array([nlf_data["frames"].index(x.split("/")[-1]) for x in frames_packed[start:end]])
        poses_nlf_init = nlf_data["poses"][nlf_inds, enum_idx].astype(np.float32)
        trans_nlf_init = nlf_data["transls"][nlf_inds, enum_idx].astype(np.float32)
        betas_nlf_init = nlf_data["betas"][:, enum_idx].copy()

    controllers_reload: Optional[list] = None
    if frame_cache_dir is None or not osp.isdir(frame_cache_dir):
        raise FileNotFoundError(
            f"getitem requires prebuilt RGBD cache, but directory is missing: {frame_cache_dir}"
        )
    try:
        input_rgbms, input_xyzs, bboxes, K_rois, poses_perturbed, frames_used, _fc, fp_indices = (
            horefine_vis_window_load_rgb_masks_depth(
                start=start,
                end=end,
                frames_packed=frames_packed,
                fp_frames=fp_frames,
                seq_name=seq_name,
                fp_poses=fp_poses,
                enum_idx=enum_idx,
                gt_to_perturb_pose=gt_to_perturb_pose,
                tar_mask_for_samples=mask_src,
                video_prefix=video_prefix,
                kid=kid,
                controllers=controllers_reload,
                kids=kids,
                render_size=render_size,
                focal=focal,
                principal_point=principal_point,
                K_full=K_full,
                fp_frame_to_idx=fp_frame_to_idx,
                frame_cache_dir=frame_cache_dir,
                mask_cache_dir=mask_cache_dir,
                tqdm_frames=False,
            )
        )

        if B_in_cams_override is not None:
            B_in_cams = np.asarray(B_in_cams_override, dtype=np.float64).copy()
        else:
            B_in_cams = np.stack(poses_perturbed, axis=0).copy()

        return horefine_vis_window_render_and_make_batch(
            B_in_cams=B_in_cams,
            verts_nlf_render=verts_nlf_render,
            prep=prep,
            trans_nlf=trans_nlf,
            poses_perturbed=poses_perturbed,
            frames_used=frames_used,
            full_colors=_fc,
            input_rgbms=input_rgbms,
            input_xyzs=input_xyzs,
            bboxes=bboxes,
            K_rois=K_rois,
            poses_full=poses_full,
            betas_gt=betas_gt,
            packed=packed,
            start=start,
            end=end,
            mesh_tensors=mesh_tensors,
            mesh_tensors_obj=mesh_tensors_obj,
            glctx=glctx,
            render_size=render_size,
            verts_obj_base=verts_obj_base,
            device=device,
            args=args,
            cfg=cfg,
            mesh_diameter=mesh_diameter,
            w2c_rots=w2c_rots,
            w2c_trans=w2c_trans,
            enum_idx=enum_idx,
            seq_name=seq_name,
            kid=kid,
            vis_input=vis_input,
            vw_input=vw_input,
            append_vis_input=bool(vis_input and record_indie_vis and vw_input is not None),
            nlf_inds=nlf_inds,
            poses_nlf_init=poses_nlf_init,
            trans_nlf_init=trans_nlf_init,
            betas_nlf_init=betas_nlf_init,
            fp_indices=fp_indices,
            render_cache_dir=render_cache_dir,
        )
    finally:
        if controllers_reload is not None:
            for _c in controllers_reload:
                _close = getattr(_c, "close", None)
                if callable(_close):
                    _close()


class HoRefineVisBatchLoader:
    """Stack identical window dicts (``batch_size``) and rebuild windows from ``ctx`` (no read from main ``batch``).

    Pass one of:
    - ``ctx=`` (namespace-like object),
    - ``ctx_ref=`` (mutable ``dict``),
    - direct context fields (e.g. ``args=..., cfg=..., device=...``).

    With ``ctx_ref``, insert the final namespace with ``finalize_horefine_vis_ctx(ctx)`` after building it.
    """

    def __init__(
        self,
        batch_size: int = 1,
        *,
        ctx: Optional[Any] = None,
        ctx_ref: Optional[dict] = None,
        **ctx_fields: Any,
    ) -> None:
        has_direct_fields = len(ctx_fields) > 0
        selected = int(ctx is not None) + int(ctx_ref is not None) + int(has_direct_fields)
        if selected != 1:
            raise ValueError(
                "HoRefineVisBatchLoader: pass exactly one of ctx=, ctx_ref=, or direct ctx fields"
            )
        self.batch_size = max(1, int(batch_size))
        self._vis_ctx = SimpleNamespace(**ctx_fields) if has_direct_fields else ctx
        self._ctx_ref = ctx_ref
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._B_override: Optional[np.ndarray] = None
        self._prep_override: Optional[dict] = None
        self._verts_override: Optional[np.ndarray] = None
        self._iter_B_in_cams: Optional[np.ndarray] = None
        self._iter_prep: Optional[dict] = None
        self._iter_verts_hum: Optional[np.ndarray] = None
        self._spent = False

    def _resolve_ctx(self) -> Any:
        if self._ctx_ref is not None:
            c = self._ctx_ref.get("ctx")
            if c is None:
                raise RuntimeError(
                    "HoRefineVisBatchLoader: ctx_ref has no 'ctx' yet; call finalize_horefine_vis_ctx(horefine_vis_ctx)"
                )
            return hydrate_horefine_vis_ctx(c)
        if self._vis_ctx is None:
            raise RuntimeError("HoRefineVisBatchLoader: missing ctx")
        return hydrate_horefine_vis_ctx(self._vis_ctx)

    def frames_packed_len(self) -> int:
        """Total number of valid packed frames for this sequence."""
        c = self._resolve_ctx()
        frames_packed = getattr(c, "frames_packed", None)
        if frames_packed is None:
            raise RuntimeError("HoRefineVisBatchLoader: ctx has no frames_packed")
        return int(len(frames_packed))

    def stack_identical_copies(self, window: dict) -> dict:
        B = self.batch_size
        if B == 1:
            return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in window.items()}

        out: dict[str, Any] = {}
        for k, v in window.items():
            if not torch.is_tensor(v):
                out[k] = v
                continue
            copies = [v.clone() for _ in range(B)]
            if v.ndim > 0 and v.shape[0] == 1:
                out[k] = torch.cat(copies, dim=0)
            else:
                out[k] = torch.stack(copies, dim=0)
        return out

    @staticmethod
    def stack_batch_list(batches: list[dict]) -> dict:
        if not batches:
            raise ValueError("empty batches")
        keys = batches[0].keys()
        out: dict[str, Any] = {}
        for k in keys:
            vals = [b[k] for b in batches]
            if torch.is_tensor(vals[0]):
                if vals[0].shape[0] == 1:
                    out[k] = torch.cat(vals, dim=0)
                else:
                    out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals[0]
        return out

    def arm(
        self,
        start: int,
        end: int,
        *,
        B_in_cams_override: Optional[np.ndarray] = None,
        prep_override: Optional[dict] = None,
        verts_hum_override: Optional[np.ndarray] = None,
    ) -> HoRefineVisBatchLoader:
        self._start = start
        self._end = end
        self._B_override = B_in_cams_override
        self._prep_override = prep_override
        self._verts_override = verts_hum_override
        self._spent = False
        return self

    def set_iter_inputs(
        self,
        *,
        B_in_cams: Optional[np.ndarray],
        prep: Optional[dict],
        verts_hum: Optional[np.ndarray],
    ) -> None:
        self._iter_B_in_cams = B_in_cams
        self._iter_prep = prep
        self._iter_verts_hum = verts_hum

    def getitem(
        self,
        start: int,
        end: int,
        it: int = 0,
    ) -> dict:
        B_in_cams_override: Optional[np.ndarray] = None
        prep_override: Optional[dict] = None
        verts_hum_override: Optional[np.ndarray] = None
        if int(it) > 0 and (
            B_in_cams_override is None or prep_override is None or verts_hum_override is None
        ):
            if B_in_cams_override is None:
                B_in_cams_override = self._iter_B_in_cams
            if prep_override is None:
                prep_override = self._iter_prep
            if verts_hum_override is None:
                verts_hum_override = self._iter_verts_hum
        _ctx = self._resolve_ctx()
        single = horefine_vis_rebuild_independent_window_batch(
            _ctx,
            start,
            end,
            B_in_cams_override=B_in_cams_override,
            prep_override=prep_override,
            verts_hum_override=verts_hum_override,
        )
        # Keep iterative state inside loader so caller can use getitem(start, end, it=it) only.
        pose_pert = single.get("obj_pose_init")
        if torch.is_tensor(pose_pert) and pose_pert.ndim >= 3:
            self._iter_B_in_cams = pose_pert[0].detach().cpu().numpy().copy()
        prep_keys = ("joints_nlf", "nlf_rotmat", "nlf_transl", "betas_gt", "betas_nlf", "nlf_poses")
        cached_prep: dict[str, Any] = {}
        for k in prep_keys:
            if k in single:
                v = single[k]
                cached_prep[k] = v.clone() if torch.is_tensor(v) else v
        if cached_prep:
            self._iter_prep = cached_prep
            try:
                self._iter_verts_hum = _horefine_verts_from_prep(_ctx, cached_prep)
            except Exception:
                self._iter_verts_hum = None
        return self.stack_identical_copies(single)
