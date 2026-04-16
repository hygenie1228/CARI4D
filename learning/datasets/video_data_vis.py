# HoRefine viz: batch stacking + independent window rebuild (mirrors ``run_1seq`` batch construction).

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import Utils
from behave_data.behave_video import load_masks
from behave_data.utils import init_video_controllers
from tools import img_utils


def horefine_vis_window_build_prep_and_smpl_init(
    runner: Any,
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
    betas = nlf_data["betas"][:, enum_idx].copy()
    betas_avg = np.mean(betas, axis=0)[None].repeat(len(poses_nlf_init), axis=0)
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
    }


def horefine_vis_window_load_rgb_masks_depth(
    runner: Any,
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
    controllers: list,
    kids: list,
    render_size: tuple,
    tqdm_frames: bool = False,
) -> tuple[list, list, list, list, list, list, list]:
    input_rgbms, input_xyzs, bboxes = [], [], []
    poses_perturbed: list = []
    K_rois: list = []
    frames_used: list = []
    full_colors: list = []
    itr = tqdm(range(start, end)) if tqdm_frames else range(start, end)
    for i in itr:
        frame_time = frames_packed[i]
        if frame_time not in fp_frames:
            print(f"Frame {frame_time} not found in FP frames!")
            continue
        idx_fp = fp_frames.index(frame_time)
        frames_used.append(f"{seq_name}/{frame_time}")
        pose_fp = np.matmul(fp_poses[idx_fp, enum_idx], gt_to_perturb_pose)
        if callable(getattr(tar_mask_for_samples, "get_masks", None)):
            mask_h, mask_o = tar_mask_for_samples.get_masks(frame_time)
        else:
            mask_h, mask_o = load_masks(video_prefix, frame_time, kid, tar_mask_for_samples)
        if mask_h is None:
            continue
        bmin, bmax = img_utils.masks2bbox([mask_h, mask_o])
        center_2d = (bmax + bmin) / 2
        radius = np.max(bmax - bmin) * 1.1 / 2
        top_left = center_2d - radius
        bottom_right = center_2d + radius
        K_roi = runner.Kroi_from_corners(bottom_right, top_left)
        t = float(frame_time[1:])
        actual_times = np.array([controllers[x].get_closest_time(t) for x, _ in enumerate(kids)])
        best_kid = np.argmin(np.abs(actual_times - t))
        actual_time = actual_times[best_kid]
        color, depth = controllers[enum_idx].get_closest_frame(actual_time)
        full_colors.append(np.asarray(color))
        color_np = np.asarray(color, dtype=np.uint8)
        mask_h_np = mask_h.astype(np.uint8)
        mask_o_np = mask_o.astype(np.uint8)
        color = np.concatenate([color_np, mask_h_np[:, :, None], mask_o_np[:, :, None]], axis=-1)
        bbox = np.hstack((top_left.astype(np.float32), bottom_right.astype(np.float32)))
        dmap_xyz, rgbm = runner.crop_color_dmap(bbox, color, depth, render_size)
        input_rgbms.append(rgbm)
        input_xyzs.append(dmap_xyz)
        K_rois.append(K_roi)
        poses_perturbed.append(pose_fp.copy())
        bboxes.append(bbox)
    return input_rgbms, input_xyzs, bboxes, K_rois, poses_perturbed, frames_used, full_colors


def horefine_vis_window_render_and_make_batch(
    runner: Any,
    *,
    B_in_cams: np.ndarray,
    verts_nlf_render: np.ndarray,
    prep: dict,
    trans_nlf: np.ndarray,
    poses_perturbed: list,
    frames_used: list,
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
    trainer: Any,
    device: str,
    args: Any,
    mesh_diameter: float,
    w2c_rots: list,
    w2c_trans: list,
    enum_idx: int,
    seq_name: str,
    kid: int,
    vis_input: bool,
    vw_input: Any,
    append_vis_input: bool,
) -> dict:
    verts_obj_batch = [
        np.matmul(verts_obj_base, pose_fp[:3, :3].T) + pose_fp[:3, 3] for pose_fp in B_in_cams
    ]
    verts_hum_batch = verts_nlf_render.copy()
    input_rgbs_final, input_xyz_final = [], []
    render_rgbs, render_xyz = [], []
    for fi in range(len(verts_obj_batch)):
        vh = verts_hum_batch[fi]
        vo = verts_obj_batch[fi]
        mesh_tensors["pos"] = torch.from_numpy(np.concatenate([vh, vo], 0)).float().cuda()
        mesh_tensors_obj["pos"] = torch.from_numpy(vo).float().cuda()
        bbox2d_ori = torch.tensor([[0, 0.0, render_size[0], render_size[1]]], device=device).repeat(1, 1)
        extra: dict = {}
        rgb_r, depth_r, _ = Utils.nvdiffrast_render(
            K=np.stack([K_rois[fi]], 0),
            H=render_size[1],
            W=render_size[0],
            ob_in_cams=torch.as_tensor(np.eye(4)[None]).float(),
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
            ob_in_cams=torch.as_tensor(np.eye(4)[None]).float(),
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
        dmap_xyz, dmap_xyz_a, rgb = trainer.dataset_test.process_input(
            dmap_xyz_init.clone(),
            fi,
            input_data,
            mesh_diameter,
            trans_nlf,
            poses_perturbed[fi],
            render_data,
            render_data["rgba"],
            frames_used[-1],
            kid,
        )
        render_rgbs.append(rgb_render.copy().transpose(2, 0, 1) / 255.0)
        render_xyz.append(dmap_xyz_a)
        input_xyz_final.append(dmap_xyz)
        input_rgbs_final.append(rgb)

    poseA_norm = B_in_cams.copy()
    poseA_norm[:, :3, 3] *= 2 / mesh_diameter
    mesh_diam_tensor = torch.as_tensor(
        [mesh_diameter] * len(poses_perturbed), device=device, dtype=torch.float
    )[None]
    trans_norm = torch.as_tensor(np.array(args.trans_normalizer), device=device, dtype=torch.float).repeat(
        len(poses_perturbed), 1
    )[None]
    batch = {
        **prep,
        "input_rgbs": torch.stack(input_rgbs_final, 0).cuda().float()[None],
        "render_rgbs": torch.from_numpy(np.stack(render_rgbs, axis=0)).cuda().float()[None],
        "input_xyz": torch.stack(input_xyz_final, 0).float().cuda()[None],
        "render_xyz": torch.stack(render_xyz, 0).float().cuda()[None],
        "mesh_diameter": mesh_diam_tensor,
        "trans_normalizer": trans_norm.reshape(1, len(poses_perturbed), 3),
        "poseA_norm": torch.from_numpy(poseA_norm).float().cuda()[None],
        "pose_perturbed": torch.from_numpy(B_in_cams)[None].float().cuda(),
        "delta_transl": torch.zeros((1, len(poses_perturbed), 3), device=device),
        "delta_rot": torch.zeros((1, len(poses_perturbed), 3, 3), device=device),
        "K_rois": torch.from_numpy(np.stack(K_rois)).float().cuda()[None],
        "smpl_poses_gt": torch.from_numpy(poses_full).float().cuda()[None],
        "smpl_transl_gt": torch.from_numpy(packed["trans"][start:end]).float().cuda()[None],
        "betas_gt": torch.from_numpy(betas_gt).float().cuda()[None],
    }
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
    batch["pose_gt"] = pose_gt_mat[None].clone()
    poseA = batch["pose_perturbed"]
    poseB = batch["pose_gt"]
    batch["delta_transl"] = poseB[:, :, :3, 3] - poseA[:, :, :3, 3]
    batch["delta_rot"] = torch.matmul(poseB[:, :, :3, :3], poseA[:, :, :3, :3].permute(0, 1, 3, 2))
    return batch


def horefine_vis_rebuild_independent_window_batch(
    runner: Any,
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
    trainer = ctx.trainer
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
    vis_input = ctx.vis_input
    vw_input = ctx.vw_input
    record_indie_vis = getattr(ctx, "record_independent_vis", False)

    # Prefer ctx.tar_mask_independent (fresh mp4 sequential reader); see ``run_horefine`` horefine_vis_ctx.
    mask_src = getattr(ctx, "tar_mask_independent", tar_mask)

    if prep_override is None:
        w = horefine_vis_window_build_prep_and_smpl_init(
            runner,
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
    else:
        prep = prep_override
        poses_full = packed["poses"][start:end].astype(np.float32)
        betas_gt = packed["betas"][start:end].astype(np.float32)
        if verts_hum_override is None:
            raise ValueError("verts_hum_override required when prep_override is set")
        verts_nlf_render = verts_hum_override.copy()
        trans_nlf = prep["nlf_transl"][0].detach().cpu().numpy()

    # Kinect readers are forward-only; main ``run_1seq`` already advanced ``ctx.controllers``.
    # Mirror ``prepare_video_mask_loader`` / ``init_video_controllers`` with fresh instances (same as reopening videos).
    video_in = getattr(args, "video", None)
    if not video_in:
        raise RuntimeError("horefine_vis: args.video is required to reload RGB/depth for independent batch")
    controllers_reload, _ = init_video_controllers(args, video_in, kids)
    try:
        input_rgbms, input_xyzs, bboxes, K_rois, poses_perturbed, frames_used, _fc = (
            horefine_vis_window_load_rgb_masks_depth(
                runner,
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
                tqdm_frames=False,
            )
        )

        if B_in_cams_override is not None:
            B_in_cams = np.asarray(B_in_cams_override, dtype=np.float64).copy()
        else:
            B_in_cams = np.stack(poses_perturbed, axis=0).copy()

        return horefine_vis_window_render_and_make_batch(
            runner,
            B_in_cams=B_in_cams,
            verts_nlf_render=verts_nlf_render,
            prep=prep,
            trans_nlf=trans_nlf,
            poses_perturbed=poses_perturbed,
            frames_used=frames_used,
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
            trainer=trainer,
            device=device,
            args=args,
            mesh_diameter=mesh_diameter,
            w2c_rots=w2c_rots,
            w2c_trans=w2c_trans,
            enum_idx=enum_idx,
            seq_name=seq_name,
            kid=kid,
            vis_input=vis_input,
            vw_input=vw_input,
            append_vis_input=bool(vis_input and record_indie_vis and vw_input is not None),
        )
    finally:
        for _c in controllers_reload:
            _close = getattr(_c, "close", None)
            if callable(_close):
                _close()


class HoRefineVisBatchLoader:
    """Stack identical window dicts (``batch_size``) and rebuild windows from ``ctx`` (no read from main ``batch``)."""

    def __init__(self, batch_size: int = 1, *, ctx: Optional[Any] = None) -> None:
        self.batch_size = max(1, int(batch_size))
        self._vis_ctx = ctx
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._B_override: Optional[np.ndarray] = None
        self._prep_override: Optional[dict] = None
        self._verts_override: Optional[np.ndarray] = None
        self._spent = False

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

    def getitem(
        self,
        start: int,
        end: int,
        *,
        B_in_cams_override: Optional[np.ndarray] = None,
        prep_override: Optional[dict] = None,
        verts_hum_override: Optional[np.ndarray] = None,
    ) -> dict:
        if self._vis_ctx is None:
            raise RuntimeError("HoRefineVisBatchLoader.getitem requires ctx=... at construction")
        single = horefine_vis_rebuild_independent_window_batch(
            self._vis_ctx.runner,
            self._vis_ctx,
            start,
            end,
            B_in_cams_override=B_in_cams_override,
            prep_override=prep_override,
            verts_hum_override=verts_hum_override,
        )
        return self.stack_identical_copies(single)

    def __iter__(self) -> HoRefineVisBatchLoader:
        self._spent = False
        return self

    def __next__(self) -> dict:
        if self._spent:
            raise StopIteration
        if self._start is None or self._end is None:
            raise RuntimeError("HoRefineVisBatchLoader: call arm(start, end) before iter")
        self._spent = True
        return self.getitem(
            self._start,
            self._end,
            B_in_cams_override=self._B_override,
            prep_override=self._prep_override,
            verts_hum_override=self._verts_override,
        )
