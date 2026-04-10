#!/usr/bin/env python3
"""Compare CoCoNet-style tensors: finetune `VideoDataset` vs a one-frame live recompute.

The live branch mirrors `HORefineRunner` + `VideoDataset.process_input` with `wild_video=False`
and `use_sel_view=False` (same defaults as `scripts/finetune.py` render prep and finetune dataloader).

Writes under --out_dir (default: tempfile):
  - finetune_sample.npz / demo_style_sample.npz (when available)
  - report.txt

Finetune tensors come from `render.h5` on disk; the live branch recomputes. They should match
after a fresh `finetune.py --prepare_only` with the same code. Note: `run_demo.py` rebuilds NLF/FP
from *init* npz under a temp tree, while this script uses `data/nlf` and `data/fp` inside `--exp_dir`.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import re
import sys
import tempfile
from types import SimpleNamespace

import h5py
import joblib
import numpy as np
import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_exp_dir(exp_dir: str) -> tuple[str, int]:
    base = osp.basename(exp_dir.rstrip("/"))
    m = re.match(r"^(.+)_(\d+)$", base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


def _to_np(x: object) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_group(path: str, flat: dict[str, np.ndarray]) -> None:
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **flat)


def _cari4d_release_cfg():
    from omegaconf import OmegaConf

    from learning.training.training_config import TrainTemporalRefinerConfig

    base = OmegaConf.structured(TrainTemporalRefinerConfig)
    yml = osp.join(ROOT, "learning/configs/cari4d-release.yml")
    return OmegaConf.merge(base, OmegaConf.load(yml))


def _finetune_sample(exp_dir: str, clip_len: int, window: int) -> dict[str, np.ndarray]:
    from omegaconf import OmegaConf

    from learning.datasets import get_dataset

    cfg = _cari4d_release_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "split_file": osp.join(exp_dir, "data", "splits", "finetune_split.json"),
                "render_root": osp.join(exp_dir, "data", "render"),
                "packed_root": osp.join(exp_dir, "data", "packed"),
                "fp_root": osp.join(exp_dir, "data", "fp"),
                "nlf_root": osp.join(exp_dir, "data", "nlf"),
                "rgb_root": "unused",
                "job": "test",
                "cam_id": parse_exp_dir(exp_dir)[1],
                "clip_len": clip_len,
                "window": window,
                "batch_size": 1,
                "num_workers": 0,
                "use_sel_view": False,
                "exclude_frames": None,
            }
        ),
    )
    _, _, _, ds_train = get_dataset(cfg)
    if len(ds_train) == 0:
        raise RuntimeError("finetune VideoDataset is empty (check render.h5 / split / clip_len).")
    sample = ds_train[0]
    keys = [
        "input_rgbs",
        "render_rgbs",
        "input_xyz",
        "render_xyz",
        "pose_perturbed",
        "pose_gt",
        "K_rois",
        "mesh_diameter",
    ]
    out: dict[str, np.ndarray] = {}
    for k in keys:
        if k not in sample:
            continue
        out[k] = _to_np(sample[k]).astype(np.float32, copy=False)
    return out


def _demo_style_frame0(exp_dir: str, rend_size: int) -> dict[str, np.ndarray]:
    """First frame of first clip, mirroring `run_horefine.run_1seq` (MP4 masks + depth video + live render)."""
    import nvdiffrast.torch as dr
    from lib_smpl import get_smpl

    import Utils
    from behave_data.const import _sub_gender
    from behave_data.utils import init_video_controllers
    from run_horefine import HORefineRunner, MP4MaskLoader
    from tools import img_utils
    from Utils import load_smpl_obj_uvmap

    video_prefix, cam_id = parse_exp_dir(exp_dir)
    seq_name = video_prefix
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("demo-style path needs CUDA (nvdiffrast + SMPL).")

    tmp = tempfile.mkdtemp(prefix="coconet_cmp_video_")
    color_mp4 = osp.join(tmp, f"{video_prefix}.{cam_id}.color.mp4")
    depth_mp4 = osp.join(tmp, f"{video_prefix}.{cam_id}.depth-reg.mp4")
    color_pkl = osp.join(tmp, f"{video_prefix}.{cam_id}.color.pkl")
    for src, dst in (
        (osp.join(exp_dir, "video.mp4"), color_mp4),
        (osp.join(exp_dir, "processed", "depth.mp4"), depth_mp4),
    ):
        if not osp.isfile(src):
            raise FileNotFoundError(f"missing {src}")
        if osp.lexists(dst):
            os.remove(dst)
        os.symlink(osp.abspath(src), dst)

    z = np.load(osp.join(exp_dir, "human", "human_params_gt.npz"), allow_pickle=True)
    intr = np.asarray(z["intrinsics"], dtype=np.float64).reshape(-1)
    joblib.dump(
        {"fx": float(intr[0]), "fy": float(intr[1]), "cx": float(intr[2]), "cy": float(intr[3])},
        color_pkl,
    )

    args = SimpleNamespace(
        video=color_mp4,
        wild_video=False,
        data_source="behave",
        rend_size=rend_size,
        nodepth=False,
        fps=30.0,
    )
    runner = HORefineRunner(args)
    kids = [cam_id]
    controllers, _ = init_video_controllers(args, color_mp4, kids)
    enum_idx = 0
    kid = cam_id

    human_m = osp.join(exp_dir, "processed", "human_mask.mp4")
    object_m = osp.join(exp_dir, "processed", "object_mask.mp4")
    tar_mask = MP4MaskLoader(human_m, object_m, fps=float(args.fps))

    fp_root = osp.join(exp_dir, "data", "fp")
    nlf_root = osp.join(exp_dir, "data", "nlf")
    fp_data = joblib.load(osp.join(fp_root, f"{seq_name}_all.pkl"))
    fp_poses = fp_data["fp_poses"]
    fp_frames = fp_data["frames"]
    nlf_data = joblib.load(osp.join(nlf_root, f"{video_prefix}_params.pkl"))
    mesh_path = osp.join(exp_dir, "object", "model.obj")
    if not osp.isfile(mesh_path):
        raise FileNotFoundError(f"demo-style compare needs HY3D mesh: {mesh_path}")
    mesh_tensors, meshes = load_smpl_obj_uvmap(seq_name, use_hy3d=True, meshes_root=mesh_path)
    meshes_any: object = meshes
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

    center = np.zeros(3, dtype=np.float64)
    gt_to_perturb_pose = np.eye(4, dtype=np.float64)
    gt_to_perturb_pose[:3, 3] = center
    verts_obj_base = verts_obj_base_t.detach().cpu().numpy() - center

    frame_time = fp_frames[0]
    idx_fp = fp_frames.index(frame_time)
    pose_fp = np.matmul(fp_poses[idx_fp, enum_idx], gt_to_perturb_pose).astype(np.float64)

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
    color_cat = np.concatenate([color_np, mask_h_np[:, :, None], mask_o_np[:, :, None]], axis=-1)
    bbox = np.hstack((top_left.astype(np.float32), bottom_right.astype(np.float32)))
    render_size = (rend_size, rend_size)
    dmap_xyz, rgbm = runner.crop_color_dmap(bbox, color_cat, depth, render_size)

    nlf_inds = np.array([nlf_data["frames"].index(fp_frames[0])])
    poses_nlf_init = nlf_data["poses"][nlf_inds, 0].astype(np.float32)
    trans_nlf_init = nlf_data["transls"][nlf_inds, 0].astype(np.float32)
    betas_gt = nlf_data["betas"][nlf_inds, 0].astype(np.float32)
    body_model = get_smpl(_sub_gender[video_prefix.split("_")[1]], hands=True).to(device)
    verts_nlf = body_model(
        torch.from_numpy(poses_nlf_init).to(device),
        torch.from_numpy(betas_gt).to(device),
        torch.from_numpy(trans_nlf_init).to(device),
    )[0].cpu().numpy()

    vo = np.matmul(verts_obj_base, pose_fp[:3, :3].T) + pose_fp[:3, 3]
    vh = verts_nlf[0]
    glctx = dr.RasterizeCudaContext()
    mesh_tensors["pos"] = torch.from_numpy(np.concatenate([vh, vo], 0)).float().cuda()
    mesh_tensors_obj["pos"] = torch.from_numpy(vo).float().cuda()
    bbox2d_ori = torch.tensor([[0, 0.0, float(render_size[0]), float(render_size[1])]], device=device)
    K_batch = np.stack([K_roi.astype(np.float32)], axis=0)
    rgb_r, depth_r, _ = Utils.nvdiffrast_render(
        K=K_batch,
        H=render_size[1],
        W=render_size[0],
        ob_in_cams=torch.as_tensor(np.eye(4)[None]).float().to(device),
        context="cuda",
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        output_size=render_size,
        bbox2d=bbox2d_ori,
        use_light=True,
    )
    rgb_obj, depth_obj, _ = Utils.nvdiffrast_render(
        K=K_batch,
        H=render_size[1],
        W=render_size[0],
        ob_in_cams=torch.as_tensor(np.eye(4)[None]).float().to(device),
        context="cuda",
        glctx=glctx,
        mesh_tensors=mesh_tensors_obj,
        output_size=render_size,
        bbox2d=bbox2d_ori,
        use_light=True,
    )
    rgbs = (rgb_r.cpu().numpy() * 255).astype(np.uint8)
    dmaps = depth_r.cpu().numpy()
    dmap_full = dmaps[0]
    dmap_obj = depth_obj[0].cpu().numpy()
    mask_rend_o = (dmap_obj <= dmap_full) & (dmap_obj > 0)
    mask_o_full = dmap_obj > 0
    dmap_xyz_init = torch.from_numpy(Utils.depth2xyzmap(dmap_full, K_roi)).permute(2, 0, 1).float()

    rgbm_np = rgbm.detach().cpu().numpy() if torch.is_tensor(rgbm) else np.asarray(rgbm)
    dxyz_np = dmap_xyz.detach().cpu().numpy() if torch.is_tensor(dmap_xyz) else np.asarray(dmap_xyz)
    input_data = {
        "rgbmB": np.clip(rgbm_np, 0, 255).astype(np.uint8).copy(),
        "xyzB": dxyz_np.astype(np.float16).copy(),
    }
    render_data = {
        "rgba": rgbs[0].copy(),
        "depth": dmaps[0].astype(np.float16).copy(),
        "K_roi": K_roi,
        "bbox": bbox,
        "mask_o": np.stack([mask_rend_o, mask_o_full], -1).copy(),
    }

    from omegaconf import OmegaConf

    from learning.datasets.video_data import VideoDataset

    cfg = _cari4d_release_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "render_root": osp.join(exp_dir, "data", "render"),
                "packed_root": osp.join(exp_dir, "data", "packed"),
                "fp_root": fp_root,
                "nlf_root": nlf_root,
                "rgb_root": "unused",
                "job": "test",
                "cam_id": cam_id,
                "use_sel_view": False,
                "exclude_frames": None,
            }
        ),
    )
    render_h5_path = osp.join(exp_dir, "data", "render", f"{seq_name}_render.h5")
    with h5py.File(render_h5_path, "r") as h5:
        mesh_diameter = float(pickle.loads(h5[f"{seq_name}_w2c"][()])["mesh_diameter"])
    ds = VideoDataset(cfg, [seq_name], "val")
    trans_nlf = trans_nlf_init
    dmap_xyz_p, dmap_xyz_a, rgb_in = ds.process_input(
        dmap_xyz_init.clone(),
        0,
        input_data,
        mesh_diameter,
        trans_nlf,
        pose_fp.astype(np.float32),
        render_data,
        render_data["rgba"],
        f"{seq_name}/{frame_time}",
        kid,
    )

    out = {
        "input_rgbs": _to_np(rgb_in).astype(np.float32),
        "render_rgbs": _to_np(render_data["rgba"].transpose(2, 0, 1) / 255.0).astype(np.float32),
        "input_xyz": _to_np(dmap_xyz_p).astype(np.float32),
        "render_xyz": _to_np(dmap_xyz_a).astype(np.float32),
        "pose_perturbed": pose_fp.astype(np.float32),
        "K_rois": K_roi.astype(np.float32),
        "mesh_diameter": np.array([mesh_diameter], dtype=np.float32),
    }
    return out


def _finetune_time0(fin: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Finetune batch uses time dimension T first; demo dump is one frame."""
    out: dict[str, np.ndarray] = {}
    for k, v in fin.items():
        if k in ("input_rgbs", "render_rgbs", "input_xyz", "render_xyz"):
            out[k] = np.asarray(v[0])
        elif k in ("pose_perturbed", "pose_gt", "K_rois"):
            out[k] = np.asarray(v[0])
        elif k == "mesh_diameter":
            out[k] = np.array([float(np.asarray(v).reshape(-1)[0])], dtype=np.float32)
        else:
            out[k] = np.asarray(v)
    return out


def _compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray], report_lines: list[str]) -> None:
    common = sorted(set(a.keys()) & set(b.keys()))
    report_lines.append(f"common keys: {common}")
    for k in common:
        xa, xb = a[k], b[k]
        if xa.shape != xb.shape:
            report_lines.append(f"  {k}: SHAPE {xa.shape} vs {xb.shape}")
            continue
        d = np.abs(xa.astype(np.float64) - xb.astype(np.float64))
        report_lines.append(
            f"  {k}: shape={xa.shape} max_abs={float(d.max()):.6g} mean_abs={float(d.mean()):.6g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--clip_len", type=int, default=32)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--rend_size", type=int, default=224)
    parser.add_argument("--demo_only", action="store_true", help="only run demo-style dump (skip finetune dataset)")
    parser.add_argument("--finetune_only", action="store_true", help="only run finetune dataset dump")
    args = parser.parse_args()

    exp_dir = osp.abspath(args.exp_dir)
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="coconet_input_cmp_")
    out_dir = osp.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    report_path = osp.join(out_dir, "report.txt")
    lines: list[str] = [f"exp_dir={exp_dir}", ""]

    fin_path = osp.join(out_dir, "finetune_sample.npz")
    demo_path = osp.join(out_dir, "demo_style_sample.npz")

    fin: dict[str, np.ndarray] = {}
    demo: dict[str, np.ndarray] = {}

    if not args.demo_only:
        fin = _finetune_sample(exp_dir, args.clip_len, args.window)
        _save_group(fin_path, fin)
        lines.append(f"wrote {fin_path}")

    if not args.finetune_only:
        demo = _demo_style_frame0(exp_dir, args.rend_size)
        _save_group(demo_path, demo)
        lines.append(f"wrote {demo_path}")

    lines.append("")
    if fin and demo:
        lines.append("=== compare (finetune clip t=0 vs demo frame0, same names only) ===")
        _compare(_finetune_time0(fin), demo, lines)
    elif fin:
        lines.append("(demo-style dump skipped or failed — no compare)")
    elif demo:
        lines.append("(finetune dump skipped — no compare)")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
