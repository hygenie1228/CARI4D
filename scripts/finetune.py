#!/usr/bin/env python3
"""Prepare one-sequence finetuning data and launch existing trainer.

This script reuses the repository training stack (`learning/training/trainer.py`)
with minimal custom glue code:
1) Build training assets under `<exp_dir>/data` from GT npz files.
2) Write `render.h5` like `run_demo.py` → `run_horefine.py`: mask-based crop, `processed/depth.mp4` for input
   XYZ, SMPL+object nvdiffrast in the ROI. Requires `object/model.obj`, depth video, and CUDA.
3) Launch trainer starting from a base checkpoint. Scheduled epochs in `--viz_epochs` export
   `finetuning_{input,output}_epochNNN.mp4` inside the trainer process (same train-dataloader batch),
   via env `FINETUNE_EXP_DIR` / `FINETUNE_VIZ_EPOCHS` — no per-epoch `run_horefine.py` subprocess.
"""

from __future__ import annotations

import argparse
import math
import shutil
import json
import os
import os.path as osp
import pickle
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import cv2
import h5py
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from omegaconf import OmegaConf


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib_smpl import SMPL_ASSETS_ROOT, get_smpl  # noqa: E402
from lib_smpl.body_landmark import BodyLandmarks  # noqa: E402


@dataclass
class PreparedPaths:
    seq_name: str
    split_json: str
    render_root: str
    packed_root: str
    nlf_root: str
    fp_root: str


def parse_exp_dir(exp_dir: str) -> tuple[str, int]:
    base = osp.basename(exp_dir.rstrip("/"))
    m = re.match(r"^(.+)_(\d+)$", base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


def _parse_epoch_list(spec: str) -> list[int]:
    if not spec.strip():
        return []
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(tok)
        if val <= 0:
            raise ValueError(f"epoch values must be positive, got {val}")
        out.append(val)
    return sorted(set(out))


def _load_cli_defaults_from_config(config_pth: str) -> dict[str, object]:
    defaults: dict[str, object] = {
        "base_ckpt": "data/base_checkpoints/base_coconet.pth",
        "start_frame": None,
        "end_frame": None,
        "clip_len": 32,
        "window": 16,
        "num_epochs": 1,
        "viz_epochs": "16,32,64,128,256,384,512",
    }
    if not osp.isfile(config_pth):
        return defaults
    try:
        cfg = OmegaConf.load(config_pth)
    except Exception as e:
        print(f"[config] warning: failed to load {config_pth}: {e}; using CLI hardcoded defaults.")
        return defaults

    def _cfg_get(key: str, fallback: object) -> object:
        val = OmegaConf.select(cfg, key, default=fallback)
        return fallback if val is None else val

    defaults["base_ckpt"] = str(_cfg_get("base_ckpt", defaults["base_ckpt"]))
    defaults["start_frame"] = _cfg_get("start_frame", defaults["start_frame"])
    defaults["end_frame"] = _cfg_get("end_frame", defaults["end_frame"])
    defaults["clip_len"] = int(_cfg_get("clip_len", defaults["clip_len"]))
    defaults["window"] = int(_cfg_get("window", defaults["window"]))
    defaults["num_epochs"] = int(_cfg_get("num_epochs", defaults["num_epochs"]))
    viz_cfg = _cfg_get("viz_epochs", defaults["viz_epochs"])
    if isinstance(viz_cfg, (list, tuple)):
        defaults["viz_epochs"] = ",".join(str(int(v)) for v in viz_cfg)
    else:
        defaults["viz_epochs"] = str(viz_cfg)
    return defaults


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    return R.from_rotvec(rotvec.astype(np.float64)).as_matrix().astype(np.float32)


def _min_frame_count(
    video_path: str,
    mask_h_path: str,
    mask_o_path: str,
    cap: int | None,
    depth_path: str | None = None,
) -> int:
    def _count(path: str) -> int:
        cap_v = cv2.VideoCapture(path)
        try:
            n = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
            return max(0, n)
        finally:
            cap_v.release()

    counts = [_count(video_path), _count(mask_h_path), _count(mask_o_path)]
    if depth_path is not None and osp.isfile(depth_path):
        counts.append(_count(depth_path))
    n = min(*counts) if cap is None else min(*counts, cap)
    if n <= 0:
        raise RuntimeError("failed to determine positive frame count from input videos")
    return n


def _load_human_gt(human_gt_npz: str, n_frames: int) -> dict:
    z = np.load(human_gt_npz, allow_pickle=True)
    required = ["global_orient", "body_pose", "lhand_pose", "rhand_pose", "betas", "trans", "intrinsics"]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"missing keys in {human_gt_npz}: {missing}")

    go = np.asarray(z["global_orient"], dtype=np.float32)[:n_frames]
    bp = np.asarray(z["body_pose"], dtype=np.float32)[:n_frames]
    lh = np.asarray(z["lhand_pose"], dtype=np.float32)[:n_frames]
    rh = np.asarray(z["rhand_pose"], dtype=np.float32)[:n_frames]
    betas = np.asarray(z["betas"], dtype=np.float32)[:n_frames]
    trans = np.asarray(z["trans"], dtype=np.float32)[:n_frames]
    intr = np.asarray(z["intrinsics"], dtype=np.float64).reshape(-1)
    if intr.size < 4:
        raise ValueError(f"invalid intrinsics in {human_gt_npz}: {intr.shape}")
    if min(go.shape[0], bp.shape[0], lh.shape[0], rh.shape[0], betas.shape[0], trans.shape[0]) < n_frames:
        raise ValueError("human GT has fewer frames than requested")

    poses = np.concatenate([go, bp, lh, rh], axis=1).astype(np.float32)  # (T, 156)
    gender = str(z["gender"].item()) if "gender" in z else "neutral"
    return {
        "poses": poses,
        "betas": betas.astype(np.float32),
        "trans": trans.astype(np.float32),
        "intrinsics": intr[:4].astype(np.float64),
        "gender": gender,
    }


def _slice_human_gt(data: dict, start_frame: int, n_frames: int) -> dict:
    end = start_frame + n_frames
    if data["poses"].shape[0] < end:
        raise ValueError(f"human GT has fewer frames than requested slice: end={end}, total={data['poses'].shape[0]}")
    return {
        "poses": data["poses"][start_frame:end].copy(),
        "betas": data["betas"][start_frame:end].copy(),
        "trans": data["trans"][start_frame:end].copy(),
        "intrinsics": data["intrinsics"].copy(),
        "gender": data["gender"],
    }


def _load_object_gt(object_gt_npz: str, n_frames: int) -> dict:
    z = np.load(object_gt_npz, allow_pickle=True)
    required = ["angle", "trans"]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"missing keys in {object_gt_npz}: {missing}")
    angle = np.asarray(z["angle"], dtype=np.float32)[:n_frames]
    trans = np.asarray(z["trans"], dtype=np.float32)[:n_frames]
    if angle.shape != trans.shape or angle.ndim != 2 or angle.shape[1] != 3:
        raise ValueError(f"invalid object GT shapes angle={angle.shape}, trans={trans.shape}")
    return {"angle": angle, "trans": trans}


def _slice_object_gt(data: dict, start_frame: int, n_frames: int) -> dict:
    end = start_frame + n_frames
    if data["angle"].shape[0] < end:
        raise ValueError(f"object GT has fewer frames than requested slice: end={end}, total={data['angle'].shape[0]}")
    return {
        "angle": data["angle"][start_frame:end].copy(),
        "trans": data["trans"][start_frame:end].copy(),
    }


def _compute_human_joints(poses: np.ndarray, betas: np.ndarray, trans: np.ndarray, gender: str) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl = get_smpl(gender if gender in ("male", "female") else "neutral", hands=True).to(device)
    landmark = BodyLandmarks(SMPL_ASSETS_ROOT)
    joints_body_list: list[np.ndarray] = []
    joints_smpl_list: list[np.ndarray] = []
    bs = 128
    with torch.no_grad():
        for s in range(0, poses.shape[0], bs):
            e = min(s + bs, poses.shape[0])
            p = torch.from_numpy(poses[s:e]).to(device)
            b = torch.from_numpy(betas[s:e]).to(device)
            t = torch.from_numpy(trans[s:e]).to(device)
            verts, joints_smpl, *_ = smpl(p, b, t)
            verts_np = verts.detach().cpu().numpy()
            joints_smpl_np = joints_smpl.detach().cpu().numpy()
            joints_body_np = landmark.get_body_kpts_batch(verts_np).astype(np.float32)
            joints_body_list.append(joints_body_np)
            joints_smpl_list.append(joints_smpl_np.astype(np.float32))
    return np.concatenate(joints_body_list, axis=0), np.concatenate(joints_smpl_list, axis=0)


def _write_support_files(
    exp_dir: str,
    seq_name: str,
    human_init: dict,
    obj_init: dict,
    human_sup: dict,
    obj_sup: dict,
    n_frames: int,
    frame_start: int = 0,
) -> PreparedPaths:
    data_root = osp.join(exp_dir, "data")
    render_root = osp.join(data_root, "render")
    packed_root = osp.join(data_root, "packed")
    nlf_root = osp.join(data_root, "nlf")
    fp_root = osp.join(data_root, "fp")
    splits_root = osp.join(data_root, "splits")
    for d in (render_root, packed_root, nlf_root, fp_root, splits_root):
        os.makedirs(d, exist_ok=True)

    frames = [f"{frame_start + i:06d}" for i in range(n_frames)]
    # NLF/FP initial states should come from init npz.
    nlf = {
        "poses": human_init["poses"][:, None, :].astype(np.float32),
        "betas": human_init["betas"][:, None, :].astype(np.float32),
        "transls": human_init["trans"][:, None, :].astype(np.float32),
        "center_pts": np.zeros((n_frames, 1, 3), dtype=np.float32),
        "center_verts": np.zeros((n_frames, 1, 3), dtype=np.float32),
        "frames": frames,
        "gender": human_init["gender"],
        "kids": [0],
    }
    joblib.dump(nlf, osp.join(nlf_root, f"{seq_name}_params.pkl"))

    # FP initialization from object init npz.
    fp_poses = np.repeat(np.eye(4, dtype=np.float32)[None, None, :, :], n_frames, axis=0)
    for i in range(n_frames):
        fp_poses[i, 0, :3, :3] = _rotvec_to_matrix(obj_init["angle"][i])
        fp_poses[i, 0, :3, 3] = obj_init["trans"][i]
    fp = {
        "fp_poses": fp_poses,
        "frames": frames,
        "fp_poses_all": fp_poses.copy(),
        "fp_best": np.zeros((n_frames, 1), dtype=np.float32),
        "visibility": np.ones((n_frames, 1), dtype=np.float32),
        "vis_thres": 0.0,
    }
    joblib.dump(fp, osp.join(fp_root, f"{seq_name}_all.pkl"))

    # Packed supervision should stay on GT labels for training loss.
    joints_body, joints_smpl = _compute_human_joints(
        human_sup["poses"], human_sup["betas"], human_sup["trans"], human_sup["gender"]
    )
    packed = {
        "obj_angles": obj_sup["angle"].astype(np.float32),
        "obj_trans": obj_sup["trans"].astype(np.float32),
        "poses": human_sup["poses"].astype(np.float32),
        "trans": human_sup["trans"].astype(np.float32),
        "betas": human_sup["betas"].astype(np.float32),
        "joints_body": joints_body.astype(np.float32),
        "joints_smpl": joints_smpl.astype(np.float32),
        "frames": frames,
        "occ_ratios": np.ones((n_frames, 4), dtype=np.float32),
        "dists_h2o": np.zeros((1, n_frames, 52), dtype=np.float32),
    }
    joblib.dump(packed, osp.join(packed_root, f"{seq_name}_GT-packed.pkl"))

    split_json = osp.join(splits_root, "finetune_split.json")
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump({"train": [seq_name], "test": [seq_name]}, f, indent=2)

    return PreparedPaths(
        seq_name=seq_name,
        split_json=split_json,
        render_root=render_root,
        packed_root=packed_root,
        nlf_root=nlf_root,
        fp_root=fp_root,
    )


def _write_render_h5_simple(
    exp_dir: str,
    paths: PreparedPaths,
    fp_all_pkl: str,
    intrinsics: np.ndarray,
    n_frames: int,
    input_size: int,
    frame_start: int = 0,
) -> None:
    """Legacy: use RGB video as synthetic `render` and flat depth (no mesh)."""
    video_path = osp.join(exp_dir, "video.mp4")
    human_mask_path = osp.join(exp_dir, "processed", "human_mask.mp4")
    object_mask_path = osp.join(exp_dir, "processed", "object_mask.mp4")
    fp = joblib.load(fp_all_pkl)
    fp_poses = fp["fp_poses"]
    frames = fp["frames"]
    out_h5 = osp.join(paths.render_root, f"{paths.seq_name}_render.h5")
    if osp.isfile(out_h5):
        os.remove(out_h5)

    cap_rgb = cv2.VideoCapture(video_path)
    cap_h = cv2.VideoCapture(human_mask_path)
    cap_o = cv2.VideoCapture(object_mask_path)
    ok0, rgb0 = cap_rgb.read()
    if not ok0 or rgb0 is None:
        raise RuntimeError(f"failed to read first frame from {video_path}")
    h0, w0 = rgb0.shape[:2]
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, float(frame_start))
    cap_h.set(cv2.CAP_PROP_POS_FRAMES, float(frame_start))
    cap_o.set(cv2.CAP_PROP_POS_FRAMES, float(frame_start))
    sx = float(input_size) / float(w0)
    sy = float(input_size) / float(h0)
    fx, fy, cx, cy = [float(x) for x in intrinsics[:4]]
    K_roi = np.array(
        [
            [fx * sx, 0.0, cx * sx],
            [0.0, fy * sy, cy * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with h5py.File(out_h5, "w") as h5:
        w2c = {
            "rot": [np.eye(3, dtype=np.float32)],
            "trans": [np.zeros(3, dtype=np.float32)],
            "mesh_diameter": float(2.0),
            "trans_normalizer": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            "rot_normalizer": float(20.0),
        }
        h5.create_dataset(f"{paths.seq_name}_w2c", data=np.void(pickle.dumps(w2c, protocol=0)))
        for i in tqdm(range(n_frames), desc="build render.h5 (simple)"):
            rr, rgb = cap_rgb.read()
            rh, mh = cap_h.read()
            ro, mo = cap_o.read()
            if not rr or not rh or not ro or rgb is None or mh is None or mo is None:
                raise RuntimeError(f"input videos ended early at frame {i}/{n_frames}")

            rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            mh = cv2.resize(cv2.cvtColor(mh, cv2.COLOR_BGR2GRAY), (input_size, input_size), interpolation=cv2.INTER_NEAREST)
            mo = cv2.resize(cv2.cvtColor(mo, cv2.COLOR_BGR2GRAY), (input_size, input_size), interpolation=cv2.INTER_NEAREST)
            mask_h = mh > 127
            mask_o = mo > 127
            fg = mask_h | mask_o

            xyz = np.zeros((input_size, input_size, 3), dtype=np.float32)
            xyz[..., 2][fg] = 1.0
            rgbm = np.concatenate(
                [
                    rgb.astype(np.uint8),
                    (mask_h.astype(np.uint8) * 255)[..., None],
                    (mask_o.astype(np.uint8) * 255)[..., None],
                ],
                axis=-1,
            ).astype(np.uint8)

            depth = xyz[..., 2].astype(np.float32)
            render_data = {
                "rgba": rgb.astype(np.uint8),
                "depth": depth,
                "pose": fp_poses[i, 0].astype(np.float32),
                "K_roi": K_roi.astype(np.float32),
                "bbox": np.array([0.0, 0.0, float(input_size - 1), float(input_size - 1)], dtype=np.float32),
                "mask_o": np.stack([mask_o, mask_o], axis=-1),
            }
            input_data = {"rgbmB": rgbm, "xyzB": xyz.astype(np.float32)}
            frame_key = f"{paths.seq_name}+{frames[i]}"
            h5.create_dataset(
                f"{frame_key}_k0_perturb_0",
                data=np.void(pickle.dumps(render_data, protocol=0)),
            )
            h5.create_dataset(
                f"{frame_key}_k0_input",
                data=np.void(pickle.dumps(input_data, protocol=0)),
            )
    cap_rgb.release()
    cap_h.release()
    cap_o.release()


def _write_render_h5_run_demo_style(
    exp_dir: str,
    paths: PreparedPaths,
    fp_all_pkl: str,
    n_frames: int,
    input_size: int,
    frame_start: int = 0,
) -> None:
    """Match `run_demo.py` / `run_horefine.run_1seq` per-frame packaging (mask crop, depth video, SMPL+obj render)."""
    import shutil
    import tempfile
    from types import SimpleNamespace

    import nvdiffrast.torch as dr

    import Utils
    from behave_data.const import _sub_gender
    from lib_smpl import get_smpl
    from run_horefine import HORefineRunner
    from tools import img_utils

    if not torch.cuda.is_available():
        raise RuntimeError("run_demo-style prepare needs CUDA (nvdiffrast).")

    seq_name = paths.seq_name
    video_prefix, cam_id = parse_exp_dir(exp_dir)
    if video_prefix != seq_name:
        print(f"[prepare] warning: exp_dir basename suggests prefix {video_prefix} != seq {seq_name}")

    hy3d = osp.join(exp_dir, "object", "model.obj")
    if not osp.isfile(hy3d):
        raise FileNotFoundError(
            f"run_demo-style finetune needs HY3D mesh at {hy3d} (same as run_demo.py)."
        )
    depth_mp4 = osp.join(exp_dir, "processed", "depth.mp4")
    if not osp.isfile(depth_mp4):
        raise FileNotFoundError(
            f"run_demo-style finetune needs depth video {depth_mp4} (same as run_demo.py)."
        )

    video_path = osp.join(exp_dir, "video.mp4")
    human_mask_path = osp.join(exp_dir, "processed", "human_mask.mp4")
    object_mask_path = osp.join(exp_dir, "processed", "object_mask.mp4")
    nlf_path = osp.join(paths.nlf_root, f"{seq_name}_params.pkl")
    fp = joblib.load(fp_all_pkl)
    fp_poses = fp["fp_poses"]
    frames = fp["frames"]
    nlf_data = joblib.load(nlf_path)

    out_h5 = osp.join(paths.render_root, f"{paths.seq_name}_render.h5")
    if osp.isfile(out_h5):
        os.remove(out_h5)

    tmpdir = tempfile.mkdtemp(prefix="finetune_behave_video_")
    try:
        color_mp4 = osp.join(tmpdir, f"{video_prefix}.{cam_id}.color.mp4")
        depth_reg = osp.join(tmpdir, f"{video_prefix}.{cam_id}.depth-reg.mp4")
        if osp.lexists(color_mp4):
            os.remove(color_mp4)
        if osp.lexists(depth_reg):
            os.remove(depth_reg)
        os.symlink(osp.abspath(video_path), color_mp4)
        os.symlink(osp.abspath(depth_mp4), depth_reg)

        # Same defaults as `run_horefine.main()` → BehaveRenderer K_full (kinect intrinsics, wild_video=False).
        args = SimpleNamespace(
            video=color_mp4,
            wild_video=False,
            data_source="behave",
            rend_size=input_size,
            nodepth=False,
            fps=30.0,
        )
        runner = HORefineRunner(args)
        kids = [cam_id]
        cfg_masks = SimpleNamespace(masks_root=f"{osp.abspath(human_mask_path)},{osp.abspath(object_mask_path)}")
        controllers, tar_mask = runner.prepare_video_mask_loader(args, kids, video_prefix, cfg_masks)

        mesh_tensors, meshes = Utils.load_smpl_obj_uvmap(
            seq_name, use_hy3d=True, meshes_root=hy3d
        )
        device = "cuda"
        meshes_any = meshes
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

        gt_to_perturb_pose = np.eye(4, dtype=np.float64)
        verts_all = verts_obj_base_t.detach().cpu().numpy()
        # Match run_demo after center_obj_bbox_inplace: poses in fp/npz are for file vertices;
        # subtracting the mesh centroid here and adding c @ R.T to translation gives the same
        # camera-space geometry as rendering centered verts with the original 4x4 pose.
        obj_mesh_centroid = verts_all.mean(axis=0).astype(np.float64)
        verts_obj_base = verts_all - obj_mesh_centroid

        sub_id = seq_name.split("_")[1]
        body_model = get_smpl(_sub_gender[sub_id], hands=True).to(device)
        betas_avg = np.mean(nlf_data["betas"][:, 0].reshape(-1, 10), axis=0)
        mesh_diameter = float(runner.get_smpl_diameter(betas_avg, body_model))

        glctx = dr.RasterizeCudaContext()
        render_size = (input_size, input_size)
        enum_idx = 0
        kid = cam_id
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        with h5py.File(out_h5, "w") as h5:
            w2c = {
                "rot": [np.eye(3, dtype=np.float32)],
                "trans": [np.zeros(3, dtype=np.float32)],
                "mesh_diameter": mesh_diameter,
                "trans_normalizer": np.array([0.5, 0.5, 0.5], dtype=np.float32),
                "rot_normalizer": float(20.0),
            }
            h5.create_dataset(f"{paths.seq_name}_w2c", data=np.void(pickle.dumps(w2c, protocol=0)))
            for i in tqdm(range(n_frames), desc="build render.h5 (run_demo-style)"):
                frame_time = frames[i]
                if frame_time not in fp["frames"]:
                    raise RuntimeError(f"frame {frame_time} missing from fp pickle")
                idx_fp = fp["frames"].index(frame_time)
                pose_fp = np.matmul(fp_poses[idx_fp, enum_idx], gt_to_perturb_pose)

                mask_h, mask_o = tar_mask.get_masks(frame_time)
                bmin, bmax = img_utils.masks2bbox([mask_h, mask_o])
                center_2d = (bmax + bmin) / 2
                radius = np.max(bmax - bmin) * 1.1 / 2
                top_left = center_2d - radius
                bottom_right = center_2d + radius
                K_roi = runner.Kroi_from_corners(bottom_right, top_left)

                t = float(frame_time[1:])
                actual_times = np.array([controllers[x].get_closest_time(t) for x, _ in enumerate(kids)])
                actual_time = actual_times[np.argmin(np.abs(actual_times - t))]
                color, depth = controllers[enum_idx].get_closest_frame(actual_time)
                color_np = np.asarray(color, dtype=np.uint8)
                mask_h_np = mask_h.astype(np.uint8)
                mask_o_np = mask_o.astype(np.uint8)
                color_cat = np.concatenate(
                    [color_np, mask_h_np[:, :, None], mask_o_np[:, :, None]],
                    axis=-1,
                )
                bbox = np.hstack((top_left.astype(np.float32), bottom_right.astype(np.float32)))
                dmap_xyz, rgbm = runner.crop_color_dmap(bbox, color_cat, depth, render_size)

                idx_nlf = nlf_data["frames"].index(frame_time)
                poses_nlf = nlf_data["poses"][idx_nlf : idx_nlf + 1, 0]
                betas_nlf = nlf_data["betas"][idx_nlf : idx_nlf + 1, 0]
                trans_nlf = nlf_data["transls"][idx_nlf : idx_nlf + 1, 0]
                verts_nlf = body_model(
                    torch.from_numpy(poses_nlf).to(device),
                    torch.from_numpy(betas_nlf).to(device),
                    torch.from_numpy(trans_nlf).to(device),
                )[0].cpu().numpy()
                vh = verts_nlf[0]
                R_fp = pose_fp[:3, :3]
                t_fp = pose_fp[:3, 3]
                t_obj = t_fp + np.matmul(obj_mesh_centroid, R_fp.T)
                vo = np.matmul(verts_obj_base, R_fp.T) + t_obj

                mesh_tensors["pos"] = torch.from_numpy(np.concatenate([vh, vo], 0)).float().cuda()
                mesh_tensors_obj["pos"] = torch.from_numpy(vo).float().cuda()
                bbox2d_ori = torch.tensor(
                    [[0, 0.0, float(render_size[0]), float(render_size[1])]],
                    device=device,
                    dtype=torch.float,
                ).repeat(1, 1)
                K_batch = np.stack([K_roi.astype(np.float32)], axis=0)
                eye_pose = torch.as_tensor(np.eye(4)[None], dtype=torch.float, device=device)
                rgb_r, depth_r, _ = Utils.nvdiffrast_render(
                    K=K_batch,
                    H=render_size[1],
                    W=render_size[0],
                    ob_in_cams=eye_pose,
                    context="cuda",
                    get_normal=False,
                    glctx=glctx,
                    mesh_tensors=mesh_tensors,
                    output_size=render_size,
                    bbox2d=bbox2d_ori,
                    use_light=True,
                    extra={},
                )
                rgb_obj, depth_obj, _ = Utils.nvdiffrast_render(
                    K=K_batch,
                    H=render_size[1],
                    W=render_size[0],
                    ob_in_cams=eye_pose,
                    context="cuda",
                    get_normal=False,
                    glctx=glctx,
                    mesh_tensors=mesh_tensors_obj,
                    output_size=render_size,
                    bbox2d=bbox2d_ori,
                    use_light=True,
                    extra={},
                )
                rgbs = (rgb_r.cpu().numpy() * 255).astype(np.uint8)
                dmaps = depth_r.cpu().numpy()
                dmap_full = dmaps[0]
                dmap_obj = depth_obj[0].cpu().numpy()
                mask_rend_o = (dmap_obj <= dmap_full) & (dmap_obj > 0)
                mask_o_full = dmap_obj > 0

                rgbm_np = rgbm.detach().cpu().numpy() if torch.is_tensor(rgbm) else np.asarray(rgbm)
                dxyz_np = dmap_xyz.detach().cpu().numpy() if torch.is_tensor(dmap_xyz) else np.asarray(dmap_xyz)
                input_data = {
                    "rgbmB": np.clip(rgbm_np, 0, 255).astype(np.uint8).copy(),
                    "xyzB": dxyz_np.astype(np.float16).copy(),
                }
                render_data = {
                    "rgba": rgbs[0].copy(),
                    "depth": dmaps[0].astype(np.float16).copy(),
                    "pose": pose_fp.astype(np.float32),
                    "K_roi": K_roi.astype(np.float32),
                    "bbox": bbox.copy(),
                    "mask_o": np.stack([mask_rend_o, mask_o_full], -1).copy(),
                }
                frame_key = f"{paths.seq_name}+{frames[i]}"
                h5.create_dataset(
                    f"{frame_key}_k0_perturb_0",
                    data=np.void(pickle.dumps(render_data, protocol=0)),
                )
                h5.create_dataset(
                    f"{frame_key}_k0_input",
                    data=np.void(pickle.dumps(input_data, protocol=0)),
                )
    finally:
        torch.set_default_tensor_type(torch.FloatTensor)
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"[prepare] run_demo-style render.h5 (SMPL+obj, mask crop, depth); mesh_diameter={mesh_diameter:.6f}")


def _write_render_h5(
    exp_dir: str,
    paths: PreparedPaths,
    fp_all_pkl: str,
    n_frames: int,
    input_size: int,
    *,
    frame_start: int = 0,
) -> None:
    _write_render_h5_run_demo_style(
        exp_dir, paths, fp_all_pkl, n_frames, input_size, frame_start=frame_start
    )


def prepare_data(
    exp_dir: str,
    start_frame: int | None,
    end_frame: int | None,
    force_rebuild: bool = False,
) -> PreparedPaths:
    seq_name, _ = parse_exp_dir(exp_dir)
    data_root = osp.join(exp_dir, "data")
    existing_paths = PreparedPaths(
        seq_name=seq_name,
        split_json=osp.join(data_root, "splits", "finetune_split.json"),
        render_root=osp.join(data_root, "render"),
        packed_root=osp.join(data_root, "packed"),
        nlf_root=osp.join(data_root, "nlf"),
        fp_root=osp.join(data_root, "fp"),
    )
    required_prepared = [
        existing_paths.split_json,
        osp.join(existing_paths.render_root, f"{seq_name}_render.h5"),
        osp.join(existing_paths.packed_root, f"{seq_name}_GT-packed.pkl"),
        osp.join(existing_paths.nlf_root, f"{seq_name}_params.pkl"),
        osp.join(existing_paths.fp_root, f"{seq_name}_all.pkl"),
    ]
    if (not force_rebuild) and osp.isdir(data_root) and all(osp.isfile(p) for p in required_prepared):
        print(f"[prepare] existing data found at {data_root}; skip rebuild.")
        print(f"[prepare] render={osp.join(existing_paths.render_root, f'{seq_name}_render.h5')}")
        return existing_paths
    if force_rebuild:
        print("[prepare] force_rebuild=True; rebuilding prepared data.")

    video_path = osp.join(exp_dir, "video.mp4")
    human_mask_path = osp.join(exp_dir, "processed", "human_mask.mp4")
    object_mask_path = osp.join(exp_dir, "processed", "object_mask.mp4")
    depth_path = osp.join(exp_dir, "processed", "depth.mp4")
    n_available = _min_frame_count(
        video_path, human_mask_path, object_mask_path, None, depth_path
    )
    input_size = 224
    frame_start = 0 if start_frame is None else int(start_frame)
    if frame_start < 0:
        raise ValueError(f"start_frame must be >= 0, got {frame_start}")
    frame_end = (n_available - 1) if end_frame is None else int(end_frame)
    frame_end = min(frame_end, n_available - 1)
    if frame_end < frame_start:
        raise ValueError(
            f"invalid frame range: start_frame={frame_start}, end_frame={frame_end}, available={n_available}"
        )
    n_frames = frame_end - frame_start + 1
    print(
        f"[prepare] sequence={seq_name}, frame range={frame_start}..{frame_end}, "
        f"frames={n_frames} (available={n_available})"
    )

    # Initialization labels are read from default paths under exp_dir.
    human_init_npz = osp.join(exp_dir, "human", "human_params_init.npz")
    object_init_npz = osp.join(exp_dir, "object", "object_params_init.npz")
    human_init_all = _load_human_gt(human_init_npz, n_available)
    obj_init_all = _load_object_gt(object_init_npz, n_available)
    human_init = _slice_human_gt(human_init_all, frame_start, n_frames)
    obj_init = _slice_object_gt(obj_init_all, frame_start, n_frames)
    # Loss supervision must use GT labels.
    human_sup_npz = osp.join(exp_dir, "human", "human_params_gt.npz")
    object_sup_npz = osp.join(exp_dir, "object", "object_params_gt.npz")
    if not osp.isfile(human_sup_npz) or not osp.isfile(object_sup_npz):
        missing = [p for p in (human_sup_npz, object_sup_npz) if not osp.isfile(p)]
        raise FileNotFoundError(f"missing required GT supervision npz: {missing}")
    human_sup_all = _load_human_gt(human_sup_npz, n_available)
    obj_sup_all = _load_object_gt(object_sup_npz, n_available)
    human_sup = _slice_human_gt(human_sup_all, frame_start, n_frames)
    obj_sup = _slice_object_gt(obj_sup_all, frame_start, n_frames)
    print(f"[prepare] supervision labels: GT ({human_sup_npz}, {object_sup_npz})")
    paths = _write_support_files(
        exp_dir, seq_name, human_init, obj_init, human_sup, obj_sup, n_frames, frame_start=frame_start
    )
    _write_render_h5(
        exp_dir=exp_dir,
        paths=paths,
        fp_all_pkl=osp.join(paths.fp_root, f"{seq_name}_all.pkl"),
        n_frames=n_frames,
        input_size=input_size,
        frame_start=frame_start,
    )
    print(f"[prepare] render={osp.join(paths.render_root, f'{seq_name}_render.h5')}")
    return paths


def run_finetune(
    exp_dir: str,
    base_ckpt: str,
    paths: PreparedPaths,
    clip_len: int,
    window: int,
    num_epochs: int,
    config_pth: str,
    viz_epochs: list[int],
) -> None:
    def _make_model_init_checkpoint(src_ckpt: str, out_dir: str) -> str:
        """Create a finetune init checkpoint without optimizer/scheduler/history states."""
        dst_ckpt = osp.join(out_dir, "step000000_init_model_only.pth")
        try:
            ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        except Exception as ex:
            print(f"[train] warning: failed to sanitize checkpoint {src_ckpt}: {ex}")
            return src_ckpt

        if isinstance(ckpt, dict) and "model" in ckpt:
            model_state = ckpt["model"]
        elif isinstance(ckpt, dict):
            # Some checkpoints store state_dict directly.
            model_state = ckpt
        else:
            print(f"[train] warning: unsupported checkpoint format at {src_ckpt}; using as-is")
            return src_ckpt

        init_ckpt = {
            "model": model_state,
            "epoch": 0,
            "step": 0,
            "best_val": None,
        }
        torch.save(init_ckpt, dst_ckpt)
        print(f"[train] created model-only init checkpoint: {dst_ckpt}")
        return dst_ckpt

    def _estimate_steps_per_epoch() -> int:
        packed_file = osp.join(paths.packed_root, f"{paths.seq_name}_GT-packed.pkl")
        if not osp.isfile(packed_file):
            return 1
        packed = joblib.load(packed_file)
        n_frames = len(packed.get("frames", []))
        if n_frames <= 0:
            return 1
        if n_frames < clip_len:
            return 1
        windows_per_view = ((n_frames - clip_len) // window) + 1
        # VideoDataset uses one selected view in this finetune command path (job=test),
        # then __len__ multiplies by 4.
        dataset_len = max(1, windows_per_view * 4)
        cfg = OmegaConf.load(config_pth)
        bs = int(getattr(cfg, "batch_size", 1) or 1)
        return max(1, math.ceil(dataset_len / bs))

    def _write_dummy_obj(path: str) -> None:
        os.makedirs(osp.dirname(path), exist_ok=True)
        if osp.isfile(path):
            return
        # tiny tetrahedron mesh, enough for trimesh loader
        text = "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
                "",
            ]
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    obj_name = paths.seq_name.split("_")[2]
    splits_dir = osp.join(ROOT, "splits")
    data_assets_dir = osp.join(ROOT, "data", "assets")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(data_assets_dir, exist_ok=True)
    exclude_views_file = osp.join(splits_dir, "exclude-views.json")
    if not osp.isfile(exclude_views_file):
        with open(exclude_views_file, "w", encoding="utf-8") as f:
            json.dump({"exclude": []}, f, indent=2)
    procigen_gender_file = osp.join(splits_dir, "procigen-video-genders.pkl")
    if not osp.isfile(procigen_gender_file):
        joblib.dump({}, procigen_gender_file)
    symm_file = osp.join(data_assets_dir, "behave-symmetries.pkl")
    if not osp.isfile(symm_file):
        joblib.dump({obj_name: np.eye(4, dtype=np.float32)[None]}, symm_file)
    # `load_templates_all` hard-loads HODome/IMHD templates too.
    # Create minimal placeholder meshes if those datasets are absent.
    from behave_data import utils as behave_utils  # local import to avoid startup side effects

    for hodome_obj in behave_utils._hodome_objs:
        _write_dummy_obj(behave_utils.get_hodome_template_path(hodome_obj))
    for imhd_obj in behave_utils._imhd_objs:
        imhd_path = osp.join("data", "imhd2", "hy3d-texgen-simp", imhd_obj, f"{imhd_obj}_simplified_transformed.obj")
        _write_dummy_obj(imhd_path)

    save_dir = osp.join(exp_dir, "data", "finetune_ckpts")
    os.makedirs(save_dir, exist_ok=True)
    # Keep all training artifacts directly under finetune_ckpts/
    exp_name = "."
    train_exp_dir = save_dir
    os.makedirs(train_exp_dir, exist_ok=True)

    # Default behavior: resume from the latest step checkpoint if it exists.
    step_ckpts = sorted(
        [
            osp.join(train_exp_dir, name)
            for name in os.listdir(train_exp_dir)
            if re.match(r"^step\d+\.pth$", name)
        ]
    )
    ckpt_for_train = step_ckpts[-1] if step_ckpts else base_ckpt
    if step_ckpts:
        print(f"[train] resume from latest checkpoint: {ckpt_for_train}")
    else:
        ckpt_for_train = _make_model_init_checkpoint(base_ckpt, train_exp_dir)
        print(f"[train] no prior step checkpoint found; start from model-only init checkpoint: {ckpt_for_train}")

    common_cmd = [
        sys.executable,
        "learning/training/trainer.py",
        f"config={config_pth}",
        f"split_file={paths.split_json}",
        f"render_root={paths.render_root}",
        f"packed_root={paths.packed_root}",
        f"nlf_root={paths.nlf_root}",
        f"fp_root={paths.fp_root}",
        f"hy3d_meshes_root={osp.join(exp_dir, 'object', 'model.obj')}",
        "rgb_root=unused",
        f"save_dir={save_dir}",
        f"exp_name={exp_name}",
        "no_wandb=True",
        "job=test",
        "cam_id=0",
        "data_name=video-data",
        "use_sel_view=False",
        "exclude_frames=null",
        f"clip_len={clip_len}",
        f"window={window}",
        "val_at_start=False",
        "val_epoch_interval=1",
        "ckpt_interval=1000000",
        "max_step_val=1",
        "debug=0",
    ]

    steps_per_epoch = _estimate_steps_per_epoch()
    num_training_steps = max(1, steps_per_epoch * num_epochs)
    train_log_file = osp.join(save_dir, "vis", "log.txt")
    cmd = common_cmd + [
        f"ckpt_file={ckpt_for_train}",
        f"num_epochs={num_epochs}",
    ]
    env = os.environ.copy()
    env["FINETUNE_LOG_FILE"] = train_log_file
    env["FINETUNE_CHUNK_TAG"] = f"full_epochs={num_epochs},ckpt={osp.basename(ckpt_for_train)}"
    env["FINETUNE_FORCE_SCHED_STEPS"] = "1"
    viz_targets = sorted({ep for ep in viz_epochs if 1 <= ep <= num_epochs})
    # Always expose experiment paths for in-trainer HoRefine (pre_finetune log / vis_t_loss).
    # viz_targets can be empty (e.g. num_epochs=1 while viz_epochs lists only >=2); epoch-end
    # export still requires FINETUNE_VIZ_EPOCHS below.
    env["FINETUNE_EXP_DIR"] = exp_dir
    env["FINETUNE_SEQ_NAME"] = paths.seq_name
    env["FINETUNE_HY3D_MESH"] = osp.join(exp_dir, "object", "model.obj")
    if viz_targets:
        env["FINETUNE_VIZ_EPOCHS"] = ",".join(str(e) for e in viz_targets)
    print(
        f"[train] scheduler num_training_steps={num_training_steps} "
        f"(steps/epoch={steps_per_epoch}, total_epochs={num_epochs})"
    )
    print("[train] running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def export_finetune_visualization(
    exp_dir: str,
    paths: PreparedPaths,
    ckpt_file: str | None = None,
    config_pth: str = "learning/configs/coconet-finetuning.yml",
    output_suffix: str | None = None,
) -> None:
    seq_name, cam_id = parse_exp_dir(exp_dir)
    default_save_dir = osp.join(exp_dir, "data", "finetune_ckpts")
    default_exp_name = "."
    if ckpt_file is None:
        # Prefer finetune output path used by this script.
        train_exp_dir = default_save_dir
        step_ckpts = (
            sorted(
                [
                    osp.join(train_exp_dir, name)
                    for name in os.listdir(train_exp_dir)
                    if re.match(r"^step\d+\.pth$", name)
                ]
            )
            if osp.isdir(train_exp_dir)
            else []
        )
        # Backward compatibility: older runs may have used exp_dir/coconet.
        if not step_ckpts:
            legacy_train_exp_dir = osp.join(exp_dir, "coconet", f"{seq_name}-finetune")
            step_ckpts = (
                sorted(
                    [
                        osp.join(legacy_train_exp_dir, name)
                        for name in os.listdir(legacy_train_exp_dir)
                        if re.match(r"^step\d+\.pth$", name)
                    ]
                )
                if osp.isdir(legacy_train_exp_dir)
                else []
            )
        if not step_ckpts:
            print(f"[viz] skip: no step checkpoint found in {train_exp_dir}")
            return
        ckpt_file = step_ckpts[-1]

    vis_dir = osp.join(default_save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    human_mask_mp4 = osp.join(exp_dir, "processed", "human_mask.mp4")
    object_mask_mp4 = osp.join(exp_dir, "processed", "object_mask.mp4")
    depth_mp4 = osp.join(exp_dir, "processed", "depth.mp4")
    video_mp4 = osp.join(exp_dir, "video.mp4")
    hy3d_mesh = osp.join(exp_dir, "object", "model.obj")
    required = [human_mask_mp4, object_mask_mp4, video_mp4, depth_mp4, hy3d_mesh, ckpt_file]
    missing = [p for p in required if not osp.exists(p)]
    if missing:
        print("[viz] skip: missing required paths for visualization:")
        for p in missing:
            print(f" - {p}")
        return

    before_ts = time.time()
    tmp_root = osp.join(tempfile.gettempdir(), "cari4d_finetune_viz", osp.basename(exp_dir))
    videos_dir = osp.join(tmp_root, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_prefix = seq_name
    color_mp4 = osp.join(videos_dir, f"{video_prefix}.{cam_id}.color.mp4")
    depth_reg = osp.join(videos_dir, f"{video_prefix}.{cam_id}.depth-reg.mp4")
    for src, dst in ((video_mp4, color_mp4), (depth_mp4, depth_reg)):
        if osp.lexists(dst):
            os.remove(dst)
        os.symlink(osp.abspath(src), dst)

    # Force fresh visualization run: remove stale outputs that trigger skip logic.
    removed_stale = 0
    for root, _, files in os.walk(vis_dir):
        for fname in files:
            if fname.endswith(".pth") or fname.endswith(".mp4"):
                stale_path = osp.join(root, fname)
                try:
                    os.remove(stale_path)
                    removed_stale += 1
                except OSError:
                    pass
    if removed_stale > 0:
        print(f"[viz] removed {removed_stale} stale visualization files from {vis_dir}")

    # Keep TensorBoard/wandb artifacts for viz runs inside finetune experiment dir,
    # instead of defaulting to experiments/cari4d-release.
    save_dir_for_viz = default_save_dir
    exp_name_for_viz = default_exp_name
    ckpt_parent = osp.dirname(ckpt_file)
    ckpt_grandparent = osp.dirname(ckpt_parent)
    if osp.abspath(ckpt_grandparent) == osp.abspath(default_save_dir):
        save_dir_for_viz = ckpt_grandparent
        exp_name_for_viz = osp.basename(ckpt_parent)

    # cmd = [
    #     sys.executable,
    #     "run_horefine.py",
    #     f"config={config_pth}",
    #     f"split_file={paths.split_json}",
    #     "use_sel_view=True",
    #     "render_video=True",
    #     "use_intermediate=True",
    #     "data_name=test-only",
    #     f"hy3d_meshes_root={hy3d_mesh}",
    #     f"masks_root={human_mask_mp4},{object_mask_mp4}",
    #     f"packed_root={paths.packed_root}",
    #     f"fp_root={paths.fp_root}",
    #     f"nlf_root={paths.nlf_root}",
    #     f"video={color_mp4}",
    #     f"cam_id={cam_id}",
    #     f"outpath={vis_dir}",
    #     f"video_out={vis_dir}",
    #     f"ckpt_file={ckpt_file}",
    #     f"save_dir={save_dir_for_viz}",
    #     f"exp_name={exp_name_for_viz}",
    #     "no_wandb=True",
    #     "job=test-only",
    #     "identifier=_finetune",
    # ]
    # print("[viz] running command:")
    # print(" ".join(cmd))
    # try:
    #     subprocess.run(cmd, check=True, cwd=ROOT)
    # except subprocess.CalledProcessError as e:
    #     print(f"[viz] warning: visualization export failed: {e}")
    #     return

    latest_pth = None
    latest_pth_mtime = -1.0
    for root, _, files in os.walk(vis_dir):
        for fname in files:
            if not fname.endswith(".pth"):
                continue
            pth_path = osp.join(root, fname)
            mtime = osp.getmtime(pth_path)
            if mtime > latest_pth_mtime:
                latest_pth_mtime = mtime
                latest_pth = pth_path
    if latest_pth is not None:
        coconet_dir = osp.join(exp_dir, "coconet")
        os.makedirs(coconet_dir, exist_ok=True)
        pth_name = "hoi_output_finetued.pth" if output_suffix is None else f"hoi_output_finetued_{output_suffix}.pth"
        dst_pth = osp.join(coconet_dir, pth_name)
        if osp.exists(dst_pth):
            os.remove(dst_pth)
        os.replace(latest_pth, dst_pth)
        print(f"[viz] moved hoi output -> {dst_pth}")
    else:
        print(f"[viz] warning: no new pth found in {vis_dir}")

    copied = 0
    for fname in os.listdir(vis_dir):
        if not fname.endswith(".mp4"):
            continue
        src_mp4 = osp.join(vis_dir, fname)
        if osp.getmtime(src_mp4) < before_ts:
            continue
        if "_input.mp4" in fname:
            out_name = "finetuning_input.mp4" if output_suffix is None else f"finetuning_input_{output_suffix}.mp4"
            dst_mp4 =  osp.join(exp_dir, out_name)
            if osp.exists(dst_mp4):
                os.remove(dst_mp4)
            os.replace(src_mp4, dst_mp4)
            copied += 1
            print(f"[viz] renamed video -> {dst_mp4}")
        else:
            out_name = "finetuning_output.mp4" if output_suffix is None else f"finetuning_output_{output_suffix}.mp4"
            dst_mp4 =  osp.join(exp_dir, out_name)
            if osp.exists(dst_mp4):
                os.remove(dst_mp4)
            os.replace(src_mp4, dst_mp4)
            copied += 1
            print(f"[viz] renamed video -> {dst_mp4}")
    if copied == 0:
        print(f"[viz] warning: no new mp4 found in {vis_dir}")


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_pth", type=str, default="learning/configs/coconet-finetuning.yml")
    pre_args, remaining = pre_parser.parse_known_args()
    default_config_pth = osp.abspath(pre_args.config_pth)
    cli_defaults = _load_cli_defaults_from_config(default_config_pth)

    parser = argparse.ArgumentParser(description="Prepare and launch CARI4D finetuning from base checkpoint.")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=cli_defaults["start_frame"])
    parser.add_argument("--end_frame", type=int, default=cli_defaults["end_frame"])
    parser.add_argument("--force_rebuild", action="store_true", help="Rebuild prepared data even if exp_dir/data already exists.",)
    args = parser.parse_args(remaining)

    exp_dir = osp.abspath(args.exp_dir)
    base_ckpt = osp.abspath(str(cli_defaults["base_ckpt"]))
    config_pth = default_config_pth
    clip_len = int(cli_defaults["clip_len"])
    window = int(cli_defaults["window"])
    num_epochs = int(cli_defaults["num_epochs"])
    viz_epochs_spec = str(cli_defaults["viz_epochs"])

    required = [
        base_ckpt,
        config_pth,
        osp.join(exp_dir, "human", "human_params_init.npz"),  # required init input
        osp.join(exp_dir, "object", "object_params_init.npz"),  # required init input
        osp.join(exp_dir, "human", "human_params_gt.npz"),  # required GT supervision
        osp.join(exp_dir, "object", "object_params_gt.npz"),  # required GT supervision
        osp.join(exp_dir, "video.mp4"),
    ]
    missing = [p for p in required if not osp.exists(p)]
    if missing:
        print("missing required paths:", file=sys.stderr)
        for p in missing:
            print(f" - {p}", file=sys.stderr)
        sys.exit(1)

    if (args.start_frame is not None or args.end_frame is not None) and not args.force_rebuild:
        print("[prepare] start_frame/end_frame is set; enabling force_rebuild=True to apply slicing.")
    force_rebuild = False # args.force_rebuild or (num_epochs <= 0) or (args.start_frame is not None) or (args.end_frame is not None)
    paths = prepare_data(
        exp_dir=exp_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        force_rebuild=force_rebuild,
    )
    if num_epochs <= 0:
        print("[train] num_epochs<=0, skipping finetuning and exporting visualization from base checkpoint.")
        export_finetune_visualization(exp_dir=exp_dir, paths=paths, ckpt_file=base_ckpt, config_pth=config_pth)
        return
    run_finetune(
        exp_dir=exp_dir,
        base_ckpt=base_ckpt,
        paths=paths,
        clip_len=clip_len,
        window=window,
        num_epochs=num_epochs,
        config_pth=config_pth,
        viz_epochs=_parse_epoch_list(viz_epochs_spec),
    )
    viz_targets = {ep for ep in _parse_epoch_list(viz_epochs_spec) if 1 <= ep <= num_epochs}
    if num_epochs not in viz_targets:
        export_finetune_visualization(exp_dir=exp_dir, paths=paths, config_pth=config_pth)


if __name__ == "__main__":
    main()
