import json
import os, sys
import re

sys.path.append(os.getcwd())

import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio
from tqdm import tqdm
import nvdiffrast.torch as dr
from pytorch3d.renderer import look_at_view_transform
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
from typing import Any, Optional

import Utils
from Utils import load_smpl_obj_uvmap
from tools import img_utils
from behave_data.utils import init_video_controllers
from lib_smpl import get_smpl
from lib_smpl.body_landmark import BodyLandmarks
from behave_data.const import _sub_gender
from behave_data.behave_video import load_masks
from prep.render_fp_nlf import BehaveFPNLFRenderer
from tools.eval_base import ModelEvaluator
from learning.training.trainer import Trainer
from learning.datasets.video_data_vis import HoRefineVisBatchLoader
from lib_smpl import pose156to72, pose72to156, SMPL_ASSETS_ROOT
import h5py


class HORefineRunner(BehaveFPNLFRenderer):
    "refine both human and object"

    @torch.no_grad()
    def run(self, args, cfg):
        "refine both human and object of one video given by cfg.video"
        # step 1: initialize a trainer the same way as in tools/eval_base.py, this will load the model and set to eval mode
        self.cfg = cfg
        cfg.job = 'test-only'
        cfg.no_wandb = True
        trainer = Trainer(cfg)
        trainer.model.eval()
        evaluator = ModelEvaluator(cfg)
        args.cam_id = cfg.cam_id
        err_keys = ['rot', 'transl', 'mpjpe', 'v2v', 'mpjae', 'smpl_t']
        errors_all = {k: [] for k in err_keys}

        # step 2: load FP pose and NLF SMPL poses from files
        args.video = cfg.video
        self.run_1seq(args, cfg, evaluator, trainer, errors_all, [])

    @torch.no_grad()
    def run_1seq(
        self,
        args,
        cfg,
        evaluator,
        trainer,
        errors_all,
        frames_all,
        metric_batch: Optional[dict] = None,
        device: str = 'cuda',
    ):
        vis_batch_loader = HoRefineVisBatchLoader(
            args=args,
            cfg=cfg,
            device=device,
            batch_size=1,
        )

        video_prefix = osp.basename(args.video).split('.')[0]
        if video_prefix == "video":
            exp_base = osp.basename(osp.dirname(args.video.rstrip("/")))
            m = re.match(r"^(.+)_(\d+)$", exp_base)
            if m:
                video_prefix = m.group(1)
        seq_name = video_prefix

        kids = [cfg.cam_id]
        mesh_tensors, meshes = load_smpl_obj_uvmap(seq_name, use_hy3d=True, meshes_root=cfg.hy3d_meshes_root)
        glctx = dr.RasterizeCudaContext()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        obj_idx = 1
        verts_obj_base_t = meshes[obj_idx].verts_padded()[0].to(device).float()

        if not cfg.wild_video:
            center = np.zeros(3) # the mesh is already aligned with GT, hence no recentering needed
        else:
            center = np.mean(verts_obj_base_t.detach().cpu().numpy(), axis=0)  # simply the center
        print("Using center: ", center)
        verts_obj_base = verts_obj_base_t.detach().cpu().numpy() - center  # need to subtract center
        body_model = get_smpl(_sub_gender[video_prefix.split('_')[1]], hands=True).to(device)
        clip_len = int(getattr(cfg, 'clip_len', 16))
        out_root = cfg.video_out
        os.makedirs(out_root, exist_ok=True)
        out_path = osp.join(out_root, f'it{cfg.refine_iters}.mp4')
        vw = imageio.get_writer(out_path, fps=30)
        landmark = BodyLandmarks(SMPL_ASSETS_ROOT)

        # For evaluation
        keys = ['pose_abs', 'smpl_pose', 'smpl_t', 'frames', 'betas', 'verts', 'contact_logits']
        data_gt, data_pr, data_in = {k: [] for k in keys}, {k: [] for k in keys}, {k: [] for k in keys}
        self.side_view_z = None # to render side view

        # Stats for in-run comparison logging in trainer.
        loss_t_eval_vals = []
        loss_r_eval_vals = []
        loss_hum_r_vals = []
        gt_fps = []
        input_fps = []
        input_parts = []

        total_frames = vis_batch_loader.frames_packed_len()
        for start in tqdm(range(0, total_frames, clip_len)):
            end = min(start + clip_len, total_frames)
            
            batch = vis_batch_loader.getitem(start, end)
            obj_pose_init_np = batch['obj_pose_init'][0].detach().cpu().numpy()
            H_full, W_full = batch["full_hw"]
            frames_used = batch["frames_used"]
            full_colors = batch["full_colors"]
            K_rois = batch['K_rois']
            poseA = batch['obj_pose_init']
            poseB = batch['obj_pose_gt']

            # Init Params
            hum_pose_init = batch['hum_pose_init'][0].float().to(device)
            hum_betas_init = batch['hum_betas_init'][0].float().to(device)
            hum_trans_init = batch['hum_transl_init'][0].float().to(device)
            
            
            # GT params
            hum_pose_gt = batch['hum_pose_gt'][0].float().to(device)
            hum_betas_gt = batch['hum_betas_gt'][0].float().to(device)
            hum_trans_gt = batch['hum_transl_gt'][0].float().to(device)

            obj_pose_init = batch['obj_pose_init'][0].float().to(device)
            obj_pose_gt = batch['obj_pose_gt'][0].float().to(device)


            batch_model = dict(batch)
            batch_model["pose_perturbed"] = batch['obj_pose_init']
            batch_model["pose_gt"] = batch['obj_pose_gt']
            batch_model["betas_gt"] = batch['hum_betas_gt']
            batch_model["betas_nlf"] = batch['hum_betas_init']
            batch_model["nlf_poses"] = batch['hum_pose_init']

            # import pdb; pdb.set_trace() # do not remove !!!(for debugging)
            rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, output = trainer.forward_batch(batch_model, cfg,
                                                                                                trainer.model,
                                                                                                ret_dict=True,
                                                                                                vis=False)
            # Record comparison metrics/fingerprints INSIDE run_1seq.
            fm = batch.get('frame_mask', None)
            if fm is None:
                bs, ts = batch['obj_pose_init'].shape[:2]
                fm = torch.ones((bs, ts), device=trans_delta_pred.device, dtype=trans_delta_pred.dtype)
            fm = fm.unsqueeze(-1)
            bs, ts = fm.shape[:2]
            loss_t_eval_vals.append(
                float((torch.abs(trans_delta_pred - trans_delta_gt).reshape(bs, ts, -1) * fm).mean().item())
            )
            loss_func_v = F.l1_loss if 'l1' in str(cfg.loss_type) else F.mse_loss
            loss_r_eval_vals.append(
                float(
                    (
                        loss_func_v(rot, rot_delta_gt, reduction='none').reshape(bs, ts, -1) * fm
                    ).mean().item() * float(getattr(cfg, "w_rot", 1.0))
                )
            )

            if ("hum_pose" in output) and ("delta_smpl_rot" in batch_model):
                gt_delta_r = batch_model["delta_smpl_rot"][:, :, :, :, :2].reshape(-1, 24 * 6)
                hum_pose = output["hum_pose"]
                loss_hum_r_vals.append(
                    float(
                        (
                            loss_func_v(hum_pose, gt_delta_r, reduction='none').reshape(bs, ts, -1) * fm
                        ).mean().item() * float(getattr(cfg, "w_hum_rot", 1.0))
                    )
                )

            # update object pose
            prep = {}
            B_in_cams = trainer.abspose_from_relative(batch_model, cfg, poseA, rot, trans_delta_pred)

            #### Start of update SMPL pose
            pred_betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = trainer.smpl_params_from_pred(batch_model, output)
            pred_smpl_pose = pose72to156(pred_smpl_pose)

            # still use the old NLF translation
            verts_init = body_model(hum_pose_init, hum_betas_init, hum_trans_init)[0].cpu().numpy()
            verts_pred = body_model(pred_smpl_pose, pred_betas, pred_smpl_t)[0].cpu().numpy()

        
            prep['nlf_transl'] = hum_trans_init[None].float()  # matches

            # joints from landmarks
            joints_nlf_np = landmark.get_body_kpts_batch(verts_pred)  # (T, 25, 3)
            prep['joints_nlf'] = torch.from_numpy(joints_nlf_np)[None].to(device).float()
            # rotation matrices per joint
            poses_nlf = pred_smpl_pose.cpu().numpy()  # (T, 72)
            nlf_rot_np = R.from_rotvec(poses_nlf.reshape(-1, 3)).as_matrix().astype(np.float32).reshape(-1, 52, 3, 3)[:, :24]# prediction has only 24 joints 
            prep['nlf_rotmat'] = torch.from_numpy(nlf_rot_np).to(device).float()  # (BT, J, 3, 3)

            prep['betas_gt'] = hum_betas_gt[None].float()
            poses_nlf, trans_nlf = poses_nlf, prep['nlf_transl'].cpu().numpy()[0]  # TODO: update betas if needed
            #### End of update SMPL pose

            # step 7: visualize predictions by rendering SMPL + object in batch (similar to tools/viz_pred.py)
            K = self.K_full.copy()
           
            # Render at full input-image scale (no downscale) so overlays align with input frames.
            scale_ratio = 2
            K[:2] /= scale_ratio
            H, W = H_full // scale_ratio, W_full // scale_ratio

            # Build combined vertices in camera space for batch
            # Object verts from predicted pose
            obj_base_centered = verts_obj_base_t - torch.as_tensor(center, device=device, dtype=torch.float)
            R_pred = B_in_cams[0, :, :3, :3].to(device).float()  # (T, 3, 3)
            t_pred = B_in_cams[0, :, :3, 3].to(device).float()  # (T, 3)
            obj_verts_pr = torch.matmul(obj_base_centered[None].expand(end - start, -1, -1),
                                        R_pred.permute(0, 2, 1)) + t_pred[:, None]        
            verts_pr = body_model(pred_smpl_pose.to(device), hum_betas_gt.reshape(-1, 10), pred_smpl_t.to(device))[0]

            verts_comb_pr = torch.cat([verts_pr, obj_verts_pr], dim=1)  # (T, N_total, 3)

            # Convert to clip space and render batch at once
            # HERE
            mtx_front, rend_pr, rend_pr_side, view_mat = self.render_front_side(H, K, W, glctx, mesh_tensors, verts_comb_pr)

            # TODO: Render input
            verts_obj_batch = [
                np.matmul(verts_obj_base, pose_fp[:3, :3].T) + pose_fp[:3, 3]
                for pose_fp in obj_pose_init_np
            ]
            verts_comb_in = torch.from_numpy(np.concatenate([verts_init, np.stack(verts_obj_batch)], 1)).float().to(device)  # (T, N_total, 3)
            _, rend_in, rend_in_side, _ = self.render_front_side(H, K, W, glctx, mesh_tensors, verts_comb_in)

            files = frames_used
            data_pr['pose_abs'].append(B_in_cams.reshape(-1, 4, 4))
            data_pr['frames'].extend(files)
            data_gt['pose_abs'].append(poseB.reshape(-1, 4, 4))
            data_gt['frames'].extend(files)
            data_in['pose_abs'].append(poseA.reshape(-1, 4, 4))
            data_in['frames'].extend(files)

            data_pr['smpl_pose'].append(pred_smpl_pose)  # (BT, 156)
            data_pr['smpl_t'].append(pred_smpl_t)
            data_pr['betas'].append(hum_betas_gt.reshape(-1, 10))
            data_gt['smpl_pose'].append(hum_pose_gt.reshape(-1, 156))
            data_gt['smpl_t'].append(hum_trans_gt.reshape(-1, 3))
            data_gt['betas'].append(hum_betas_gt.reshape(-1, 10))
            data_pr['verts'].append(verts_pr)
            data_in['verts'].append(torch.from_numpy(verts_init).to(device).float())

            # add contact logits
            if 'contact' in output:
                data_pr['contact_logits'].append(output['contact'].reshape(-1, 2))
                data_gt['contact_logits'].append(output['contact'].reshape(-1, 2))
                data_in['contact_logits'].append(output['contact'].reshape(-1, 2))
                print("contact logits added")
            else:
                # add dummy data 
                data_pr['contact_logits'].append(torch.zeros(len(files), 2))
                data_gt['contact_logits'].append(torch.zeros(len(files), 2))
                data_in['contact_logits'].append(torch.zeros(len(files), 2))

            frames_all.extend(files) # to accumulate for all

            # Input data
            data_in['smpl_pose'].append(hum_pose_init.reshape(-1, 156))
            data_in['smpl_t'].append(hum_trans_init.reshape(-1, 3))
            betas_avg = hum_betas_gt.mean(dim=0, keepdim=True).repeat(len(hum_pose_init), 1)
            data_in['betas'].append(betas_avg.reshape(-1, 10))

            
            # Prepare GT SMPL and object for visualization
            if not cfg.wild_video:
                vs_gt_world = body_model(hum_pose_gt, hum_betas_gt, hum_trans_gt)[0]
                
                data_gt['verts'].append(vs_gt_world)

                # GT object verts are already available in camera coordinates via pose_gt.
                R_cam = poseB[0, :, :3, :3].to(device).float()
                t_cam = poseB[0, :, :3, 3].to(device).float()
                obj_verts_gt = torch.matmul(
                    obj_base_centered[None].expand(end - start, -1, -1),
                    R_cam.permute(0, 2, 1),
                ) + t_cam[:, None]
                verts_comb_gt = torch.cat([vs_gt_world, obj_verts_gt], dim=1)
                _, rend_gt, rend_gt_side, _ = self.render_front_side(H, K, W, glctx, mesh_tensors, verts_comb_gt)

            try:
                maskA, maskB, rgbsA, rgbsB, xyzA, xyzB = trainer.prepare_input_viz(batch, cfg)
                bboxes = np.asarray(batch.get("bboxes", None), dtype=np.float32) if batch.get("bboxes", None) is not None else None
                bboxes_scaled = (bboxes / float(scale_ratio)) if bboxes is not None else None

                for j in tqdm(range(end - start)):
                    frame_time = osp.basename(frames_used[j])
                    # reuse preloaded color
                    color = cv2.resize(full_colors[j], (W, H))
                    in_comb = self.comb_front_side(color, rend_in[j], rend_in_side[j])
                    pr_comb = self.comb_front_side(color, rend_pr[j], rend_pr_side[j])
                    rgb_comb = self.comb_front_side(color, color, np.ones_like(color)*127)
                    combs = [rgb_comb]
                    bid = 0

                    def _to_vis3(x: np.ndarray) -> np.ndarray:
                        if x.ndim == 2:
                            x = x[:, :, None]
                        if x.shape[2] == 1:
                            x = np.repeat(x, 3, axis=2)
                        elif x.shape[2] == 2:
                            x = np.concatenate([x, np.zeros_like(x[:, :, :1])], axis=2)
                        elif x.shape[2] > 3:
                            x = x[:, :, :3]
                        if x.dtype != np.uint8:
                            x = x.astype(np.float32)
                            if x.max() <= 1.0:
                                x = x * 255.0
                            x = np.clip(x, 0, 255).astype(np.uint8)
                        return x

                    def _uncrop_to_full(
                        img: np.ndarray,
                        bbox: np.ndarray,
                        out_h: int,
                        out_w: int,
                        interp: int,
                        bg_value: int = 0,
                    ) -> np.ndarray:
                        x1, y1, x2, y2 = [float(v) for v in bbox]
                        x1i, y1i = int(np.floor(x1)), int(np.floor(y1))
                        x2i, y2i = int(np.ceil(x2)), int(np.ceil(y2))
                        x1i = max(0, min(out_w - 1, x1i))
                        y1i = max(0, min(out_h - 1, y1i))
                        x2i = max(x1i + 1, min(out_w, x2i))
                        y2i = max(y1i + 1, min(out_h, y2i))
                        patch = cv2.resize(img, (x2i - x1i, y2i - y1i), interpolation=interp)
                        canvas = np.full((out_h, out_w, 3), int(bg_value), dtype=np.uint8)
                        canvas[y1i:y2i, x1i:x2i] = patch
                        return canvas

                    maska_vis = _to_vis3(maskA[bid, j].transpose(1, 2, 0))
                    maskb_vis = _to_vis3(maskB[bid, j].transpose(1, 2, 0))
                    if bboxes_scaled is not None and j < len(bboxes_scaled):
                        maska_vis = _uncrop_to_full(maska_vis, bboxes_scaled[j], H, W, cv2.INTER_NEAREST)
                        maskb_vis = _uncrop_to_full(maskb_vis, bboxes_scaled[j], H, W, cv2.INTER_NEAREST)
                    mask_panel = self.comb_front_side(maska_vis, maska_vis, maskb_vis)

                    xyza_vis = (np.clip(xyzA[bid, j].transpose(1, 2, 0) + 0.5, 0, 1.0) * 255).astype(np.uint8)
                    xyzb_vis = (np.clip(xyzB[bid, j].transpose(1, 2, 0) + 0.5, 0, 1.0) * 255).astype(np.uint8)
                    if bboxes_scaled is not None and j < len(bboxes_scaled):
                        # XYZ visualization uses +0.5 offset mapping, so neutral background is mid-gray.
                        xyza_vis = _uncrop_to_full(xyza_vis, bboxes_scaled[j], H, W, cv2.INTER_LINEAR, bg_value=127)
                        xyzb_vis = _uncrop_to_full(xyzb_vis, bboxes_scaled[j], H, W, cv2.INTER_LINEAR, bg_value=127)
                    xyz_panel = self.comb_front_side(xyza_vis, xyza_vis, xyzb_vis)

                    def _fit_h(img: np.ndarray, h: int) -> np.ndarray:
                        ih, iw = img.shape[:2]
                        return cv2.resize(img, (max(1, int(iw * h / ih)), h))

                    h_target = rgb_comb.shape[0]
                    combs.extend([_fit_h(mask_panel, h_target), _fit_h(xyz_panel, h_target)])

                    combs.extend([in_comb, pr_comb])
                    if not cfg.wild_video:
                        gt_comb = self.comb_front_side(color, rend_gt[j], rend_gt_side[j])
                        combs.append(gt_comb)
                        # Validate GT side-view alignment (row2 col3 vs row2 col4).
                        h0, w0 = color.shape[:2]
                        x1, x2 = int(w0 * 0.15), int(w0 * 0.85)
                        y1, y2 = int(h0 * 0.15), int(h0 * 1.0)
                        pr_side = rend_pr_side[j][y1:y2, x1:x2]
                        gt_side = rend_gt_side[j][y1:y2, x1:x2]

                    comb = np.concatenate(combs, axis=1)
                    cv2.putText(comb, frame_time+ f' idx {j+start}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 255), 4)

                    vw.append_data(comb)
            finally:
                pass
        vw.close()
        print(f'visualization saved to {out_path}')
        # save result as one pth file 
        pth_file = f'{cfg.outpath}/{seq_name}.pth'
        os.makedirs(osp.dirname(pth_file), exist_ok=True)
        data_pr = {k: torch.cat(v, 0) if k != 'frames' and len(v) > 0 else v for k, v in data_pr.items()}
        data_gt = {k: torch.cat(v, 0) if k != 'frames' and len(v) > 0 else v for k, v in data_gt.items()}
        data_in = {k: torch.cat(v, 0) if k != 'frames' and len(v) > 0 else v for k, v in data_in.items()}
        torch.save({"gt": data_gt, "pr": data_pr, "in": data_in}, pth_file)
        print(f'result saved to {pth_file}')
        # Loss/metrics from the exact prediction used right before visualization rendering.
        loss_t_viz_render = (float(loss_t_eval_vals[-1]) if len(loss_t_eval_vals) > 0 else None)
        loss_r_viz_render = (float(loss_r_eval_vals[-1]) if len(loss_r_eval_vals) > 0 else None)
        loss_hum_r_viz_render = (float(loss_hum_r_vals[-1]) if len(loss_hum_r_vals) > 0 else None)

        # Optional train-side reference recomputation on metric_batch (for debug/comparison only).
        loss_t_viz_metric_batch = None
        loss_r_viz_metric_batch = None
        loss_hum_r_viz_metric_batch = None
        metric_batch_gt_fps = []
        metric_batch_input_fps = []
        metric_batch_input_parts = []
        if metric_batch is not None:
            rot_ref, rot_delta_gt_ref, trans_delta_gt_ref, trans_delta_pred_ref, out_ref = trainer.forward_batch(
                metric_batch, cfg, trainer.model, ret_dict=True, vis=False
            )
            fm_ref = metric_batch.get("frame_mask", None)
            if fm_ref is None:
                bs_ref, ts_ref = metric_batch["pose_perturbed"].shape[:2]
                fm_ref = torch.ones(
                    (bs_ref, ts_ref), device=trans_delta_pred_ref.device, dtype=trans_delta_pred_ref.dtype
                )
            fm_ref = fm_ref.unsqueeze(-1)
            bs_ref, ts_ref = fm_ref.shape[:2]
            if hasattr(trainer, "_compute_logged_t_loss_from_forward"):
                loss_t_viz_metric_batch = float(
                    trainer._compute_logged_t_loss_from_forward(
                        metric_batch, cfg, trans_delta_gt_ref, trans_delta_pred_ref, out_ref
                    ).item()
                )
            else:
                loss_t_viz_metric_batch = float(
                    (torch.abs(trans_delta_pred_ref - trans_delta_gt_ref).reshape(bs_ref, ts_ref, -1) * fm_ref)
                    .mean()
                    .item()
                )
            loss_func_ref = F.l1_loss if 'l1' in str(cfg.loss_type) else F.mse_loss
            loss_r_viz_metric_batch = float(
                (
                    loss_func_ref(rot_ref, rot_delta_gt_ref, reduction='none').reshape(bs_ref, ts_ref, -1) * fm_ref
                ).mean().item() * float(getattr(cfg, "w_rot", 1.0))
            )
            if ("hum_pose" in out_ref) and ("delta_smpl_rot" in metric_batch):
                gt_delta_r_ref = metric_batch["delta_smpl_rot"][:, :, :, :, :2].reshape(-1, 24 * 6)
                hum_pose_ref = out_ref["hum_pose"]
                loss_hum_r_viz_metric_batch = float(
                    (
                        loss_func_ref(hum_pose_ref, gt_delta_r_ref, reduction='none').reshape(bs_ref, ts_ref, -1)
                        * fm_ref
                    ).mean().item() * float(getattr(cfg, "w_hum_rot", 1.0))
                )
            if hasattr(trainer, "coconet_gt_fingerprint"):
                metric_batch_gt_fps = [trainer.coconet_gt_fingerprint(metric_batch)]
            if hasattr(trainer, "coconet_model_input_fingerprint"):
                metric_batch_input_fps = [trainer.coconet_model_input_fingerprint(metric_batch)]
            if hasattr(trainer, "coconet_model_input_fingerprint_parts"):
                metric_batch_input_parts = [trainer.coconet_model_input_fingerprint_parts(metric_batch)]

        # Primary viz losses for trainer comparison:
        # prefer metric_batch-recomputed values (same batch as train/eval_t_loss),
        # fallback to render-path values when metric_batch is unavailable.
        loss_t_viz = loss_t_viz_metric_batch if loss_t_viz_metric_batch is not None else loss_t_viz_render
        loss_r_viz = loss_r_viz_metric_batch if loss_r_viz_metric_batch is not None else loss_r_viz_render
        loss_hum_r_viz = (
            loss_hum_r_viz_metric_batch if loss_hum_r_viz_metric_batch is not None else loss_hum_r_viz_render
        )
        self.last_run_1seq_stats = {
            "loss_t_viz": loss_t_viz,
            "loss_r_viz": loss_r_viz,
            "loss_hum_r_viz": loss_hum_r_viz,
            "loss_t_viz_render": loss_t_viz_render,
            "loss_r_viz_render": loss_r_viz_render,
            "loss_hum_r_viz_render": loss_hum_r_viz_render,
            "loss_t_viz_metric_batch": loss_t_viz_metric_batch,
            "loss_r_viz_metric_batch": loss_r_viz_metric_batch,
            "loss_hum_r_viz_metric_batch": loss_hum_r_viz_metric_batch,
            "gt_fps": gt_fps,
            "input_fps": input_fps,
            "input_parts": input_parts,
            "metric_batch_gt_fps": metric_batch_gt_fps,
            "metric_batch_input_fps": metric_batch_input_fps,
            "metric_batch_input_parts": metric_batch_input_parts,
        }
        return self.last_run_1seq_stats

    def comb_front_side(self, color, rp, rp_side):
        "rp: front view, rp_side: side view"
        mask_p = (rp.sum(axis=-1, keepdims=True) > 0)
        alpha = 0.7
        pred_top = color.copy()
        pred_top[mask_p[..., 0]] = (alpha * rp[mask_p[..., 0]] + (1 - alpha) * pred_top[mask_p[..., 0]]).astype(np.uint8)
        # cut
        if not self.cfg.wild_video:
            h, w = color.shape[:2]
            x1, x2 = int(w*0.15), int(w*0.85)
            y1, y2 = int(h*0.15), int(h*1)
            pr_comb = np.concatenate((pred_top[y1:y2, x1:x2], rp_side[y1:y2, x1:x2]), axis=0)
        else:
            # no cut
            pr_comb = np.concatenate((pred_top, rp_side), axis=0)
        return pr_comb

    def render_front_side(self, H, K, W, glctx, mesh_tensors, verts_comb_pr):
        device = verts_comb_pr.device

        projection_mat = torch.as_tensor(
            Utils.projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100).reshape(1, 4, 4),
            device=device, dtype=torch.float)
        ob_in_glcams = torch.tensor(Utils.glcam_in_cvcam, device=device, dtype=torch.float).reshape(1, 4, 4)
        mtx_front = (projection_mat @ ob_in_glcams).repeat(len(verts_comb_pr), 1, 1)  # (T, 4, 4)
        pos_homo = Utils.to_homo_torch(verts_comb_pr)
        pos_clip = (mtx_front[:, None] @ pos_homo[..., None])[..., 0]
        rend_batch = Utils.nvdiff_rasterize(glctx, mesh_tensors, pos_clip, (H, W))  # (T, H, W, 3)
        rend_batch = (rend_batch * 255).byte().cpu().numpy()
        # Side view transformation
        if self.side_view_z is None:
            self.side_view_z = torch.mean(verts_comb_pr[:, :, 2])
        z_now = torch.mean(verts_comb_pr[:, :, 2])
        if abs(z_now - self.side_view_z) > 1.0:
            self.side_view_z = z_now
        z = self.side_view_z
        
        at = torch.tensor([[0.0, 0.0, z]], device=device, dtype=torch.float)
        Rv, Tv = look_at_view_transform(dist=z*1.3, elev=0, azim=75, at=at, up=((0, 1, 0),), device=device)
        view_mat = torch.eye(4, device=device, dtype=torch.float)
        view_mat[:3, :3] = Rv[0]
        view_mat[:3, 3] = Tv[0]
        verts_pr_side = torch.matmul(verts_comb_pr, view_mat[:3, :3]) + view_mat[:3, 3]
        pos_homo_side = Utils.to_homo_torch(verts_pr_side)
        pos_clip_side = (mtx_front[:, None] @ pos_homo_side[..., None])[..., 0]
        rend_batch_side = Utils.nvdiff_rasterize(glctx, mesh_tensors, pos_clip_side, (H, W))
        rend_batch_side = (rend_batch_side * 255).byte().cpu().numpy()
        return mtx_front, rend_batch, rend_batch_side, view_mat

    def render_front_side_Ks(self, H, W, Ks, glctx, mesh_tensors, verts_comb_pr):
        "Like render_front_side but Ks is (T,3,3) intrinsics per frame (training ROI crops can differ per t)."
        device = verts_comb_pr.device
        T = len(verts_comb_pr)
        mats = [
            Utils.projection_matrix_from_intrinsics(Ks[j], height=H, width=W, znear=0.001, zfar=100).reshape(4, 4)
            for j in range(T)
        ]
        projection_mat = torch.as_tensor(np.stack(mats, axis=0), device=device, dtype=torch.float)
        ob_in_glcams = torch.tensor(Utils.glcam_in_cvcam, device=device, dtype=torch.float).reshape(1, 4, 4)
        mtx_front = projection_mat @ ob_in_glcams
        pos_homo = Utils.to_homo_torch(verts_comb_pr)
        pos_clip = (mtx_front[:, None] @ pos_homo[..., None])[..., 0]
        rend_batch = Utils.nvdiff_rasterize(glctx, mesh_tensors, pos_clip, (H, W))
        rend_batch = (rend_batch * 255).byte().cpu().numpy()
        if self.side_view_z is None:
            self.side_view_z = torch.mean(verts_comb_pr[:, :, 2])
        z_now = torch.mean(verts_comb_pr[:, :, 2])
        if abs(z_now - self.side_view_z) > 1.0:
            self.side_view_z = z_now
        z = self.side_view_z
        at = torch.tensor([[0.0, 0.0, z]], device=device, dtype=torch.float)
        Rv, Tv = look_at_view_transform(dist=z * 1.3, elev=0, azim=75, at=at, up=((0, 1, 0),), device=device)
        view_mat = torch.eye(4, device=device, dtype=torch.float)
        view_mat[:3, :3] = Rv[0]
        view_mat[:3, 3] = Tv[0]
        verts_pr_side = torch.matmul(verts_comb_pr, view_mat[:3, :3]) + view_mat[:3, 3]
        pos_homo_side = Utils.to_homo_torch(verts_pr_side)
        pos_clip_side = (mtx_front[:, None] @ pos_homo_side[..., None])[..., 0]
        rend_batch_side = Utils.nvdiff_rasterize(glctx, mesh_tensors, pos_clip_side, (H, W))
        rend_batch_side = (rend_batch_side * 255).byte().cpu().numpy()
        return mtx_front, rend_batch, rend_batch_side, view_mat


def _mesh_tensors_to_device(mesh_tensors: dict, device: torch.device) -> dict:
    "load_smpl_obj_uvmap returns CPU/cuda tensors; align with trainer device."
    out = {}
    for k, v in mesh_tensors.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def main():
    from argparse import Namespace
    from glob import glob

    from learning.training.trainer import get_config
    cfg = get_config()

    # check if the video is a path or a pattern to files 
    if osp.isfile(cfg.video):
        videos = [cfg.video]
    else:
        videos = sorted(glob(cfg.video))
    print(f'In total {len(videos)} videos')
    for video in videos:
        cfg.video = video

        args = Namespace(
            # from fp_behave.py
            video=cfg.video,
            outpath='outputs/fp',
            fps=30,
            tstart=3.0,
            tend=None,
            redo=False,
            kid=1,
            start=0,
            end=None,  # override (-1) with your snippet's default
            nodepth=False,

            # from your provided defaults
            packed_path='data/behave/behave-packed/',
            dataset_path='data/behave/',
            output_dir='outputs/foundpose_train/behave',
            h5_path='data/behave_release/30fps-h5',
            shard_num=5,
            trans_normalizer=[0.02, 0.02, 0.05],
            rot_normalizer=20.0,
            rend_size=224,
            skip=1,
            add_rgb=False,
            data_source='behave',
        )
        args.wild_video = cfg.wild_video
        args.cam_id = cfg.cam_id

        runner = HORefineRunner(args)
        runner.run(args, cfg)


if __name__ == '__main__':
    main()

