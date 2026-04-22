# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
trainer
"""
import sys, os
import json
import time
import hashlib
import shutil
import tempfile
from typing import Optional

import cv2
import trimesh
from PIL import Image

sys.path.append(os.getcwd())
import wandb
import torch
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
from accelerate import Accelerator
from learning.datasets import get_dataset
from learning.training.training_utils import TrainState, get_scheduler
from tools.geometry_utils import geodesic_distance
from learning.training.training_config import TrainTemporalRefinerConfig
from pytorch3d.transforms.so3 import so3_log_map, so3_exp_map
from torch.optim.lr_scheduler import LambdaLR
import logging
import os.path as osp
import Utils
from lib_smpl import get_smpl, pose72to156
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs
from learning.models import get_model
import tools.geometry_utils as geom_utils
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, cfg:TrainTemporalRefinerConfig):
        self.cfg = cfg
        self.freeze_bn_stats = bool(getattr(cfg, "freeze_bn_stats", False))
        self.exp_dir = osp.join(cfg.save_dir, cfg.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # --- 1. Initialize Accelerator ---
        # `Accelerator` will automatically handle device placement, gradient scaling, etc.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # fix ddp problem
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        # --- 2. Create Model, Optimizer, and Loss Function ---
        model = get_model(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.train_state = TrainState()
        loaded_scheduler_state = False
        if cfg.lr_scheduler.type == 'none':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            scheduler = get_scheduler(cfg, optimizer)
        if cfg.ckpt_file is not None:
            ckpt_files = [cfg.ckpt_file]
        else:
            # load ckpt
            ckpt_files = sorted(glob(osp.join(self.exp_dir, '*.pth')))
        if len(ckpt_files) == 0:
            assert cfg.job == 'train', 'No ckpt found, and job is not train'
            fp_ckpt_file = "experiments/weights/2023-10-28-18-33-37/model_best.pth"
            
            if cfg.use_fp_pretrained:
                ckpt = torch.load(fp_ckpt_file)
                if 'model' in ckpt:
                    ckpt = ckpt['model']
                # TODO: adapt the pose_embed.pe tensor
                ckpt_new = {}
                for k, v in ckpt.items():
                    if k in ['pos_embed.pe', 'time_pose.pe']:
                        if model.pos_embed.pe.shape != ckpt[k].shape:
                            print(f"Warning: {k} in ckpt shape {ckpt[k].shape} != {model.pos_embed.pe.shape}")
                        else:
                            ckpt_new[k] = v
                    else: ckpt_new[k] = v
                missing_keys, unexpected_keys = model.load_state_dict(ckpt_new, strict=False)
                if len(missing_keys):
                    print(f' - Missing_keys: {missing_keys}')
                if len(unexpected_keys):
                    print(f' - Unexpected_keys: {unexpected_keys}')
                print('loaded model from checkpoint', fp_ckpt_file)
            else:
                if cfg.use_fp_head:
                    states_model = model.state_dict()
                    for k, v in states_model.items():
                        if 'rot_head' in k or 'trans_head' in k:
                            states_model[k] = ckpt[k]
                            print(f'reusing the weight of {k} from FP')
                    model.load_state_dict(states_model)
                else:
                    print("Not loading any ckpt, train from scratch!")
        else:
            # load model and optimizer state
            ckpt = torch.load(ckpt_files[-1], weights_only=False)
            ckpt_new = {}
            for k, v in ckpt['model'].items():
                if k in ['pos_embed.pe', 'time_pose.pe']:
                    if model.pos_embed.pe.shape != ckpt['model'][k].shape:
                        print(f"Warning: {k} in ckpt shape {ckpt['model'][k].shape} != {model.pos_embed.pe.shape}")
                    else:
                        ckpt_new[k] = v
                else:
                    ckpt_new[k] = v
            missing_keys, unexpected_keys = model.load_state_dict(ckpt_new, strict=False)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            else:
                print("Warning: no optimizer states found in the ckpt!")
            self.train_state = TrainState(ckpt['epoch'], ckpt['step'], ckpt['best_val'])
            fp_ckpt_file = ckpt_files[-1]
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
                loaded_scheduler_state = True
            else:
                print('No scheduler states found in the ckpt!')

            print('loaded model from checkpoint', fp_ckpt_file)
        self.ckpt_file = fp_ckpt_file
        print('loss type:', cfg.loss_type)
        print("Total number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # --- 3. Create the DataLoader ---
        dataloader_train, dataloader_val, dataset_test, dataset_train = get_dataset(cfg)

        # --- 4. The magic `accelerator.prepare()` call ---
        # This wraps all our components, making them ready for distributed training.
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler = accelerator.prepare(
            model, optimizer, dataloader_train, dataloader_val, scheduler,
        )
        self.accelerator = accelerator
        self.dataset_test = dataset_test
        if self.freeze_bn_stats and accelerator.is_main_process:
            print("[train] freeze_bn_stats=True: keep BatchNorm running stats fixed during train mode.")

        # For fresh finetune starts, align scheduler horizon with actual train iterations.
        # Keep resumed scheduler state untouched when loaded from checkpoint.
        if (not loaded_scheduler_state) and cfg.lr_scheduler.type in ['transformers', 'cosine', 'cosine_with_restarts']:
            steps_per_epoch = max(1, len(self.train_dataloader))
            old_total = None
            requested_total = None
            old_warmup = None
            if hasattr(cfg.lr_scheduler, "kwargs") and "num_training_steps" in cfg.lr_scheduler.kwargs:
                try:
                    old_total = int(cfg.lr_scheduler.kwargs["num_training_steps"])
                    requested_total = old_total
                except Exception:
                    old_total = None
            if hasattr(cfg.lr_scheduler, "kwargs") and "num_warmup_steps" in cfg.lr_scheduler.kwargs:
                try:
                    old_warmup = int(cfg.lr_scheduler.kwargs["num_warmup_steps"])
                except Exception:
                    old_warmup = None
            force_sched_from_data = os.environ.get("FINETUNE_FORCE_SCHED_STEPS", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            # For finetune launcher, force scheduler horizon to real dataloader steps.
            # Otherwise, keep backward-compatible behavior that respects explicit overrides.
            if force_sched_from_data:
                total_steps = max(1, steps_per_epoch * int(cfg.num_epochs))
            else:
                total_steps = requested_total if (requested_total is not None and requested_total > 0) else max(1, steps_per_epoch * int(cfg.num_epochs))
            cfg.lr_scheduler.kwargs["num_training_steps"] = int(total_steps)
            warmup_steps = max(1, int(total_steps) // 20)
            cfg.lr_scheduler.kwargs["num_warmup_steps"] = int(warmup_steps)
            self.scheduler = get_scheduler(cfg, self.optimizer)
            print(
                f"[scheduler] set num_training_steps={total_steps} "
                f"num_warmup_steps={warmup_steps} "
                f"(steps/epoch={steps_per_epoch}, num_epochs={cfg.num_epochs}, requested={requested_total}, "
                f"old_total={old_total}, old_warmup={old_warmup}, force_from_data={force_sched_from_data})"
            )

        # init logging
        self.tb_writer = None
        if accelerator.is_main_process:
            tb_dir = osp.join(self.exp_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard logging enabled: {tb_dir}")

        if not cfg.no_wandb and accelerator.is_main_process:
            # find out the previous wandb run
            run_folders = sorted(glob(osp.join(self.exp_dir, 'wandb/run-*')))
            if len(run_folders) == 0:
                print("NO wandb run found, starting from scratch!")
                wandb.init(project=cfg.wandb_project, name=cfg.exp_name, job_type=cfg.job,
                           config=OmegaConf.to_container(cfg),
                           dir=self.exp_dir)
            else:
                print('found runs:', [osp.basename(x) for x in run_folders])
                print('resume wandb from', run_folders[-1])
                wandb.init(project=cfg.wandb_project, name=cfg.exp_name, job_type=cfg.job,
                           config=OmegaConf.to_container(cfg),
                           id=osp.basename(run_folders[-1]).split('-')[-1],
                           dir=self.exp_dir, resume='must')


        # Init human model
        if cfg.nlf_root is not None:
            self.smpl_male = get_smpl('male', True).cuda()
            self.smpl_female = get_smpl('female', True).cuda()
        self._coconet_input_history = []

    def _set_model_mode(self, training: bool) -> None:
        """Set train/eval mode, optionally freezing BN running stats in train mode."""
        self.model.train(training)
        if training and self.freeze_bn_stats:
            for mod in self.model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()

    @staticmethod
    def _tensor_fingerprint(tensor: torch.Tensor, sample_size: int = 4096) -> str:
        """Compute a stable lightweight hash for large tensor comparison."""
        # Move first to CPU to avoid any CUDA indexing kernels in debug utility code.
        flat = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        if flat.numel() == 0:
            return "empty"
        if flat.numel() > sample_size:
            step = max(1, flat.numel() // sample_size)
            flat = flat[::step][:sample_size]
        arr = flat.numpy()
        return hashlib.sha1(arr.tobytes()).hexdigest()

    @staticmethod
    def _get_poseA_norm(batch) -> torch.Tensor:
        """Return normalized init pose expected by CoCoNet."""
        if "poseA_norm" in batch:
            return batch["poseA_norm"]
        if ("obj_pose_init" in batch) and ("mesh_diameter" in batch):
            pose = batch["obj_pose_init"].clone()
            pose[:, :, :3, 3] *= 2.0 / batch["mesh_diameter"].reshape(len(pose), pose.shape[1], 1)
            return pose
        raise KeyError("missing poseA_norm (or obj_pose_init+mesh_diameter)")

    @staticmethod
    def _get_obj_delta_supervision(batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (delta_transl, delta_rot); derive from absolute poses when absent."""
        if ("delta_transl" in batch) and ("delta_rot" in batch):
            return batch["delta_transl"], batch["delta_rot"]

        pose_init = batch.get("obj_pose_init", batch.get("pose_perturbed", None))
        pose_gt = batch.get("obj_pose_gt", batch.get("pose_gt", None))
        if pose_init is None or pose_gt is None:
            raise KeyError("missing delta supervision and absolute object poses")

        delta_transl = pose_gt[:, :, :3, 3] - pose_init[:, :, :3, 3]
        delta_rot = torch.matmul(pose_gt[:, :, :3, :3], pose_init[:, :, :3, :3].permute(0, 1, 3, 2))
        return delta_transl, delta_rot

    @staticmethod
    def coconet_input_fingerprint_dict(batch, frame_ids: Optional[list] = None) -> dict:
        """JSON-serializable fingerprints for CoCONet inputs (train vs viz pipeline check)."""
        out: dict = {}
        keys = (
            "input_rgbs",
            "render_rgbs",
            "input_xyz",
            "render_xyz",
            "pose_perturbed",
            "poseA_norm",
            "K_rois",
            "delta_transl",
            "delta_rot",
            "mesh_diameter",
            "trans_normalizer",
        )
        for k in keys:
            if k not in batch:
                continue
            v = batch[k]
            if torch.is_tensor(v):
                out[k] = Trainer._tensor_fingerprint(v)
            elif isinstance(v, np.ndarray):
                out[k] = Trainer._tensor_fingerprint(torch.from_numpy(v.astype(np.float32)))
        if frame_ids is not None:
            out["frame_ids"] = [str(x) for x in frame_ids]
        elif "image_files" in batch:
            imgs = batch["image_files"]
            if torch.is_tensor(imgs):
                out["frame_ids"] = [str(x) for x in imgs.detach().cpu().numpy().tolist()]
            else:
                out["frame_ids"] = [str(x) for x in list(imgs)]
        return out

    @staticmethod
    def coconet_gt_fingerprint(batch) -> str:
        """Stable hash of GT-relevant supervision for train/viz equality checks."""
        parts: dict = {}
        keys = (
            "pose_gt",
            "delta_transl",
            "delta_rot",
            "smpl_poses_gt",
            "smpl_transl_gt",
            "betas_gt",
            "frame_mask",
        )
        for k in keys:
            if k not in batch:
                continue
            v = batch[k]
            if torch.is_tensor(v):
                parts[k] = Trainer._tensor_fingerprint(v)
            elif isinstance(v, np.ndarray):
                parts[k] = Trainer._tensor_fingerprint(torch.from_numpy(v.astype(np.float32)))
        if "image_files" in batch:
            imgs = batch["image_files"]
            if torch.is_tensor(imgs):
                frame_ids = [str(x) for x in imgs.detach().cpu().numpy().tolist()]
            else:
                frame_ids = [str(x) for x in list(imgs)]
            parts["frame_ids_sha1"] = hashlib.sha1(
                json.dumps(frame_ids, ensure_ascii=True).encode("utf-8")
            ).hexdigest()
        packed = json.dumps(parts, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(packed.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def coconet_model_input_fingerprint(batch) -> str:
        """Hash ONLY tensors directly fed to CoCoNet model.forward in forward_batch()."""
        parts = Trainer.coconet_model_input_fingerprint_parts(batch)
        packed = json.dumps(parts, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(packed.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def coconet_model_input_fingerprint_parts(batch) -> dict[str, str]:
        # forward_batch(): model(cat([render_rgbs, render_xyz], 2), cat([input_rgbs, input_xyz], 2), poseA_norm, batch)
        in_a = torch.cat([batch["render_rgbs"], batch["render_xyz"]], 2)
        in_b = torch.cat([batch["input_rgbs"], batch["input_xyz"]], 2)
        pose_p = Trainer._get_poseA_norm(batch)
        return {
            "in_a": Trainer._tensor_fingerprint(in_a),
            "in_b": Trainer._tensor_fingerprint(in_b),
            "poseA_norm": Trainer._tensor_fingerprint(pose_p),
        }

    def _log_metrics(self, log_dict: dict, step: int) -> None:
        """Log to wandb (if enabled) and TensorBoard (scalars only)."""
        if self.accelerator.is_main_process and (not self.cfg.no_wandb):
            wandb.log(log_dict, step=step)
        if not (self.accelerator.is_main_process and self.tb_writer is not None):
            return
        # Keep TensorBoard train/*, val/*, and viz/* aligned for easy side-by-side comparison.
        metric_suffixes = {
            "loss",        # total loss
            "loss_acc",
            "loss_r",      # object rotation loss
            "loss_t",      # object translation loss
            "lr",
            "loss_hum_r",
            "loss_hum_r_raw",
            "loss_hum_t",
            "loss_hum_j",
            "loss_hum_b",
            "loss_hum_velo",
            "loss_obj_velo",
            "loss_contact",
            "loss_t_abs",
            "loss_r_abs",
        }
        allowed_train_tb_keys = {f"train/{k}" for k in metric_suffixes}
        allowed_val_tb_keys = {f"val/{k}" for k in metric_suffixes}
        allowed_viz_tb_keys = {f"viz/{k}" for k in metric_suffixes}
        legacy_tb_map = {
            "loss_train": "train/loss",
            "loss_train_r": "train/loss_r",
            "loss_train_t": "train/loss_t",
            "loss_train_acc": "train/loss_acc",
            "lr": "train/lr",
            "loss_val": "val/loss",
            "loss_val_r": "val/loss_r",
            "loss_val_t": "val/loss_t",
            "loss_val_acc": "val/loss_acc",
            "loss_val_hum_r": "val/loss_hum_r",
            "loss_val_hum_t": "val/loss_hum_t",
            "loss_val_hum_j": "val/loss_hum_j",
            "loss_val_hum_b": "val/loss_hum_b",
            "loss_val_hum_velo": "val/loss_hum_velo",
            "loss_val_obj_velo": "val/loss_obj_velo",
            "loss_val_contact": "val/loss_contact",
        }
        for key, value in log_dict.items():
            scalar = None
            if torch.is_tensor(value):
                if value.numel() == 1:
                    scalar = float(value.detach().cpu().item())
            elif isinstance(value, (float, int, np.floating, np.integer)):
                scalar = float(value)
            if scalar is not None:
                tb_key = legacy_tb_map.get(str(key), str(key))
                if tb_key.startswith("train/") and tb_key not in allowed_train_tb_keys:
                    continue
                if tb_key.startswith("val/") and tb_key not in allowed_val_tb_keys:
                    continue
                if tb_key.startswith("viz/") and tb_key not in allowed_viz_tb_keys:
                    continue
                if tb_key.startswith("train/") or tb_key.startswith("val/") or tb_key.startswith("viz/"):
                    self.tb_writer.add_scalar(tb_key, scalar, step)

    def _finetune_viz_schedule(self, cfg):
        fe_dir = (getattr(cfg, "finetune_exp_dir", None) or "").strip()
        if not fe_dir:
            fe_dir = os.environ.get("FINETUNE_EXP_DIR", "").strip()
        raw_env = os.environ.get("FINETUNE_VIZ_EPOCHS", "").strip()
        if raw_env:
            epochs = sorted({int(x.strip()) for x in raw_env.split(",") if x.strip()})
        else:
            ve = getattr(cfg, "finetune_viz_epochs", None)
            if not ve:
                epochs = []
            else:
                epochs = sorted({int(x) for x in list(ve)})
        return fe_dir, epochs

    def _export_finetune_epoch_videos(
        self,
        batch,
        epoch_1based: int,
        finetune_exp_dir: str,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Write finetuning_input (simple box) and finetuning_output (HoRefine-style mesh grid when possible)."""
        if not self.accelerator.is_main_process:
            return None, None, None, None

        cfg = self.cfg
        model = self.model
        was_training = model.training
        viz_loss_t_render = None
        viz_loss_t_render2 = None
        viz_loss_r = None
        viz_loss_hum_r = None
        out_in = osp.join(finetune_exp_dir, f"finetuning_input_epoch{epoch_1based:03d}.mp4")
        out_out = osp.join(finetune_exp_dir, f"finetuning_output_epoch{epoch_1based:03d}.mp4")

        bid = 0
        seq_name = os.environ.get("FINETUNE_SEQ_NAME", "").strip()
        if not seq_name and batch.get("image_files") is not None:
            try:
                seq_name = str(batch["image_files"][bid][0]).split(os.sep)[0]
            except Exception:
                seq_name = ""
        mesh_ok = False
        if seq_name:
            try:
                from argparse import Namespace
                from run_horefine import HORefineRunner
                from tools.eval_base import ModelEvaluator
                from omegaconf import OmegaConf

                # Mirror scripts/finetune.py export_finetune_visualization command but run in-process.
                exp_dir = osp.abspath(finetune_exp_dir)
                cam_id = int(getattr(cfg, "cam_id", 0))
                video_prefix = seq_name
                videos_dir = osp.join(exp_dir, "videos")
                os.makedirs(videos_dir, exist_ok=True)

                human_mask_mp4 = osp.join(exp_dir, "processed", "human_mask.mp4")
                object_mask_mp4 = osp.join(exp_dir, "processed", "object_mask.mp4")
                depth_mp4 = osp.join(exp_dir, "processed", "depth.mp4")
                video_mp4 = osp.join(exp_dir, "video.mp4")
                color_mp4 = osp.join(videos_dir, f"{video_prefix}.{cam_id}.color.mp4")
                depth_reg = osp.join(videos_dir, f"{video_prefix}.{cam_id}.depth-reg.mp4")
                for src, dst in ((video_mp4, color_mp4), (depth_mp4, depth_reg)):
                    if osp.lexists(dst):
                        os.remove(dst)
                    os.symlink(osp.abspath(src), dst)

                # Use a temporary directory for per-epoch visualization artifacts so
                # scripts/finetune.py does not leave exp_dir/vis_epoch* directories.
                vis_dir = tempfile.mkdtemp(prefix=f"finetune_vis_epoch{epoch_1based:03d}_")
                before_ts = time.time()

                cfg_viz = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                data_source = str(getattr(cfg, "data_source", "behave"))
                wild_video = bool(getattr(cfg, "wild_video", False))
                cfg_viz.split_file = getattr(cfg, "split_file", None)
                cfg_viz.use_sel_view = True
                cfg_viz.render_video = True
                cfg_viz.use_intermediate = True
                cfg_viz.data_name = "test-only"
                cfg_viz.hy3d_meshes_root = (
                    (os.environ.get("FINETUNE_HY3D_MESH", "") or "").strip() or getattr(cfg, "hy3d_meshes_root", "")
                )
                cfg_viz.masks_root = f"{osp.abspath(human_mask_mp4)},{osp.abspath(object_mask_mp4)}"
                cfg_viz.packed_root = getattr(cfg, "packed_root", None)
                cfg_viz.fp_root = getattr(cfg, "fp_root", None)
                cfg_viz.nlf_root = getattr(cfg, "nlf_root", None)
                cfg_viz.video = color_mp4
                cfg_viz.cam_id = cam_id
                cfg_viz.outpath = vis_dir
                cfg_viz.video_out = vis_dir
                cfg_viz.no_wandb = True
                cfg_viz.job = "test-only"
                cfg_viz.identifier = "_finetune"

                args = Namespace(
                    video=cfg_viz.video,
                    outpath=cfg_viz.outpath,
                    fps=30,
                    tstart=3.0,
                    tend=None,
                    redo=False,
                    kid=1,
                    start=0,
                    end=None,
                    nodepth=False,
                    packed_path="data/behave/behave-packed/",
                    dataset_path="data/behave/",
                    output_dir="outputs/foundpose_train/behave",
                    h5_path="data/behave_release/30fps-h5",
                    shard_num=5,
                    trans_normalizer=[0.02, 0.02, 0.05],
                    rot_normalizer=20.0,
                    rend_size=224,
                    skip=1,
                    add_rgb=False,
                    data_source=data_source,
                )
                args.wild_video = wild_video
                args.cam_id = cfg_viz.cam_id

                evaluator = ModelEvaluator(cfg_viz)
                err_keys = ["rot", "transl", "mpjpe", "v2v", "mpjae", "smpl_t"]
                errors_all = {k: [] for k in err_keys}
                runner = HORefineRunner(args)
                runner.cfg = cfg_viz

                was_training2 = model.training
                # run_horefine.run_1seq sets global default tensor type to cuda float tensor.
                # Restore it after export; otherwise next DataLoader worker fork can crash with
                # "Cannot re-initialize CUDA in forked subprocess".
                _default_tensor_type_before = torch.tensor(0.0).type()
                model.eval()

                with torch.no_grad():
                    _stats = runner.run_1seq(
                        args,
                        cfg_viz,
                        evaluator,
                        self,
                        errors_all,
                        [],
                        batch,
                    )
                    if _stats is None:
                        _stats = getattr(runner, "last_run_1seq_stats", None)
                    if _stats is None:
                        raise RuntimeError("run_1seq did not return stats for viz comparison")
                    viz_loss_t_render = _stats.get("loss_t_viz_render", None)
                    viz_loss_t_render = _stats.get(
                        "loss_t_viz_render",
                        _stats.get("loss_t_viz", _stats.get("viz_loss_t_eval", None)),
                    )
                    viz_loss_t_render2 = _stats.get("loss_t_viz_render2", None)
                    viz_loss_r = _stats.get(
                        "loss_r_viz_render",
                        _stats.get("loss_r_viz", None),
                    )
                    viz_loss_hum_r = _stats.get(
                        "loss_hum_r_viz_render",
                        _stats.get("loss_hum_r_viz", None),
                    )
                if _default_tensor_type_before == "torch.FloatTensor":
                    torch.set_default_tensor_type(torch.FloatTensor)
                elif _default_tensor_type_before == "torch.cuda.FloatTensor":
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
                else:
                    # expected to be float tensors in this project; fall back to CPU float.
                    torch.set_default_tensor_type(torch.FloatTensor)
                self._set_model_mode(was_training2)

                latest_out = None
                latest_out_mtime = -1.0
                latest_in = None
                latest_in_mtime = -1.0
                for root, _, files in os.walk(vis_dir):
                    for fname in files:
                        if not fname.endswith(".mp4"):
                            continue
                        src_mp4 = osp.join(root, fname)
                        mtime = osp.getmtime(src_mp4)
                        if mtime < before_ts:
                            continue
                        if "_input.mp4" in fname:
                            if mtime > latest_in_mtime:
                                latest_in_mtime = mtime
                                latest_in = src_mp4
                        else:
                            if mtime > latest_out_mtime:
                                latest_out_mtime = mtime
                                latest_out = src_mp4

                if latest_out is not None:
                    if osp.exists(out_out):
                        os.remove(out_out)
                    shutil.copy2(latest_out, out_out)
                    mesh_ok = True
                if latest_in is not None:
                    if osp.exists(out_in):
                        os.remove(out_in)
                    shutil.copy2(latest_in, out_in)
                print(f"[finetune-viz] run_horefine in-process export done (mesh_grid={mesh_ok})")
                shutil.rmtree(vis_dir, ignore_errors=True)
            except Exception as e:
                if "vis_dir" in locals():
                    shutil.rmtree(vis_dir, ignore_errors=True)
                raise RuntimeError(f"[finetune-viz] run_horefine in-process export failed: {e}") from e
        if not mesh_ok:
            raise RuntimeError(
                "[finetune-viz] run_horefine visualization was not produced (mesh_grid=False). "
                "Fallback visualization is disabled by user request."
            )
        self._set_model_mode(was_training)
        if osp.exists(out_in):
            print(f"[finetune-viz] wrote {out_in} and {out_out} (mesh_grid={mesh_ok})")
        else:
            print(
                f"[finetune-viz] wrote {out_out} (mesh_grid={mesh_ok}); "
                "input visualization intentionally skipped"
            )
        return viz_loss_t_render, viz_loss_r, viz_loss_hum_r, viz_loss_t_render2

    def train(self):
        cfg = self.cfg
        accelerator = self.accelerator
        model, optimizer, train_dataloader, val_dataloader = self.model, self.optimizer, self.train_dataloader, self.val_dataloader
        scheduler = self.scheduler
        finetune_log_file = os.environ.get("FINETUNE_LOG_FILE", "").strip()
        finetune_chunk_tag = os.environ.get("FINETUNE_CHUNK_TAG", "").strip()

        def _append_finetune_train_log(
            reason: str, loss_t_value: float
        ) -> None:
            if (not finetune_log_file) or (not accelerator.is_main_process):
                return
            os.makedirs(osp.dirname(finetune_log_file), exist_ok=True)
            tag = f" tag={finetune_chunk_tag}" if finetune_chunk_tag else ""
            line = (
                f"train_end{tag} reason={reason} epoch={train_state.epoch} "
                f"step={train_state.step} loss_t={loss_t_value:.8f}\n"
            )
            with open(finetune_log_file, "a", encoding="utf-8") as f:
                f.write(line)
            print(f"[train] appended to {finetune_log_file}: {line.strip()}")

        def _append_finetune_viz_compare_log(
            epoch_1based: int,
            train_t_loss: Optional[float],
            vis_t_loss: Optional[float] = None,
            vis_t_loss2: Optional[float] = None,
        ) -> None:
            if (not finetune_log_file) or (not accelerator.is_main_process):
                return
            os.makedirs(osp.dirname(finetune_log_file), exist_ok=True)
            tag = f" tag={finetune_chunk_tag}" if finetune_chunk_tag else ""
            train_t_part = "nan" if train_t_loss is None else f"{train_t_loss:.8f}"
            vis_part = "nan" if vis_t_loss is None else f"{vis_t_loss:.8f}"
            vis2_part = "nan" if vis_t_loss2 is None else f"{vis_t_loss2:.8f}"
            line = (
                f"viz_compare{tag} epoch={epoch_1based} step={train_state.step} "
                f"train_t_loss={train_t_part} vis_t_loss={vis_part} vis_t_loss2={vis2_part}\n"
            )
            with open(finetune_log_file, "a", encoding="utf-8") as f:
                f.write(line)
            print(f"[train] appended viz comparison to {finetune_log_file}: {line.strip()}")

        def _append_finetune_prefinetune_log(
            train_t_loss: Optional[float],
            vis_t_loss: Optional[float] = None,
            vis_t_loss2: Optional[float] = None,
        ) -> None:
            if (not finetune_log_file) or (not accelerator.is_main_process):
                return
            os.makedirs(osp.dirname(finetune_log_file), exist_ok=True)
            tag = f" tag={finetune_chunk_tag}" if finetune_chunk_tag else ""
            train_part = "nan" if train_t_loss is None else f"{train_t_loss:.8f}"
            vis_part = "nan" if vis_t_loss is None else f"{vis_t_loss:.8f}"
            vis2_part = "nan" if vis_t_loss2 is None else f"{vis_t_loss2:.8f}"
            line = (
                f"pre_finetune{tag} epoch={train_state.epoch} step={train_state.step} "
                f"train_t_loss={train_part} vis_t_loss={vis_part} vis_t_loss2={vis2_part}\n"
            )
            with open(finetune_log_file, "a", encoding="utf-8") as f:
                f.write(line)
            print(f"[train] appended pre-finetune stats to {finetune_log_file}: {line.strip()}")

        # --- 5. The Training Loop ---
        train_state = self.train_state
        pre_finetune_logged = False
        if cfg.val_at_start:
            print('Evaluation at the start of training.')
            self.eval_model(cfg, model, train_state, val_dataloader)
        accelerator.print(f"Starting training...")
        last_loss_t_value = 0.0
        fixed_ref_batch = None
        for epoch in range(cfg.num_epochs):
            self._set_model_mode(True)
            total_loss = 0.0
            total_loss_t = 0.0
            epoch_last_batch = None

            for step, batch in enumerate(train_dataloader):
                epoch_last_batch = batch
                if fixed_ref_batch is None:
                    # Keep a fixed reference batch (first train batch) for apples-to-apples
                    # pre-finetune vs post-finetune loss/viz comparisons.
                    fixed_ref_batch = batch
                if (not pre_finetune_logged) and epoch == 0 and step == 0:
                    # Log baseline before any finetune optimizer update.
                    pre_train_t, pre_viz_t = None, None
                    pre_viz_t_render2 = None
                    with torch.no_grad():
                        self._set_model_mode(True)
                        _, _, trans_delta_gt_t0, trans_delta_pred_t0, out_dict_t0 = self.forward_batch(
                            batch, cfg, model, vis=False, ret_dict=True
                        )
                        pre_train_t = float(
                            self._compute_logged_t_loss_from_forward(
                                batch, cfg, trans_delta_gt_t0, trans_delta_pred_t0, out_dict_t0
                            ).item()
                        )
                        self._set_model_mode(True)
                    fe_dir_pref, _ = self._finetune_viz_schedule(cfg)
                    if fe_dir_pref:
                        pre_viz_t, _, _, pre_viz_t_render2 = self._export_finetune_epoch_videos(
                            fixed_ref_batch,
                            0,
                            fe_dir_pref,
                        )
                    _append_finetune_prefinetune_log(
                        pre_train_t,
                        pre_viz_t,
                        pre_viz_t_render2,
                    )
                    pre_finetune_logged = True
                # No need for .to(device), accelerate handles it!
                # Forward pass
                loss, loss_r, loss_t, loss_acc, _loss_t_train_delta_masked = self.forward_step(
                    batch, cfg, model, vis=step % cfg.vis_every_n_steps == 0
                )
                last_loss_t_value = float(loss_t.item())
                if not cfg.no_wandb and accelerator.is_main_process:
                    log_dict = {"loss_train": loss.item(), 'loss_train_r': loss_r.item(),
                                'loss_train_t': loss_t.item(), 'lr': optimizer.param_groups[0]["lr"]}
                    self._log_metrics(log_dict, train_state.step)
                elif accelerator.is_main_process:
                    # Keep the same core train loss keys in TensorBoard even when wandb is disabled.
                    self._log_metrics(
                        {
                            "loss_train": loss.item(),
                            "loss_train_r": loss_r.item(),
                            "loss_train_t": loss_t.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        train_state.step,
                    )

                # Backward pass - accelerator handles the backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                total_loss += loss.item()
                total_loss_t += loss_t.item()
                train_state.step += 1

                # Print progress from the main process only
                if accelerator.is_main_process and step % 20 == 0:
                    accelerator.print(f"Epoch [{epoch + 1}/{cfg.num_epochs}], Step [{step}], Loss: {loss.item():.4f}")

                if train_state.step % cfg.ckpt_interval == 0 and accelerator.is_main_process:
                    self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state)

                lr = optimizer.param_groups[0]['lr']
                if lr < 1e-7:
                    print("Learning rate too small, stopping training.")
                    self.eval_model(cfg, model, train_state, val_dataloader)
                    if accelerator.is_main_process:
                        self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state)
                    _append_finetune_train_log(
                        reason="lr_too_small",
                        loss_t_value=last_loss_t_value,
                    )
                    return

            # Log average loss for the epoch from the main process
            if accelerator.is_main_process:
                avg_loss = total_loss / len(train_dataloader)
                accelerator.print(f"--- End of Epoch [{epoch + 1}/{cfg.num_epochs}], Average Loss: {avg_loss:.4f} ---")
                ref_batch = fixed_ref_batch if fixed_ref_batch is not None else epoch_last_batch
                if ref_batch is not None:
                    # Recompute on the final model state (post-optimizer updates), on a fixed
                    # reference batch for apples-to-apples pre/post finetune comparison.
                    was_training_epoch_end = model.training
                    with torch.no_grad():
                        self._set_model_mode(True)
                        _, _, trans_delta_gt_t, trans_delta_pred_t, out_dict_t = self.forward_batch(
                            ref_batch, cfg, model, vis=False, ret_dict=True
                        )
                        loss_t_train_post = self._compute_logged_t_loss_from_forward(
                            ref_batch, cfg, trans_delta_gt_t, trans_delta_pred_t, out_dict_t
                        )

                    self._set_model_mode(was_training_epoch_end)
                    last_loss_t_value = float(loss_t_train_post.item())
                    accelerator.print(
                        f"--- End of Epoch [{epoch + 1}/{cfg.num_epochs}], Loss_t_train (post-update, viz-batch): "
                        f"{last_loss_t_value:.6f} ---"
                    )
            if ((epoch + 1) % int(getattr(cfg, "val_epoch_interval", 1)) == 0):
                self.eval_model(cfg, model, train_state, val_dataloader)

            fe_dir, fe_epochs = self._finetune_viz_schedule(cfg)
            completed_1based = epoch + 1
            if (
                accelerator.is_main_process
                and fe_dir
                and fe_epochs
                and completed_1based in fe_epochs
                and (fixed_ref_batch is not None or epoch_last_batch is not None)
            ):
                ref_batch = fixed_ref_batch if fixed_ref_batch is not None else epoch_last_batch
                self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state)
                step_pth = osp.join(self.exp_dir, f"step{train_state.step:06d}.pth")
                ep_pth = osp.join(self.exp_dir, f"epoch{completed_1based:03d}.pth")
                if osp.isfile(step_pth):
                    shutil.copy2(step_pth, ep_pth)
                    print(f"[finetune-viz] saved {ep_pth}")
                viz_loss_t_render, viz_loss_r, viz_loss_hum_r, viz_loss_t_render2 = self._export_finetune_epoch_videos(
                    ref_batch,
                    completed_1based,
                    fe_dir,
                )
                viz_metrics = {
                    "viz/loss_t": viz_loss_t_render if viz_loss_t_render is not None else float("nan"),
                    "viz/loss_t2": viz_loss_t_render2 if viz_loss_t_render2 is not None else float("nan"),
                    "viz/loss_r": viz_loss_r if viz_loss_r is not None else float("nan"),
                    "viz/loss_hum_r": viz_loss_hum_r if viz_loss_hum_r is not None else float("nan"),
                }
                self._log_metrics(viz_metrics, train_state.step)
                # Ensure viz scalars are emitted even if higher-level filtering/mapping changes.
                if self.accelerator.is_main_process and self.tb_writer is not None:
                    for _k, _v in viz_metrics.items():
                        try:
                            _f = float(_v)
                        except Exception:
                            continue
                        if np.isfinite(_f):
                            self.tb_writer.add_scalar(_k, _f, train_state.step)
                    self.tb_writer.flush()
                _append_finetune_viz_compare_log(
                    epoch_1based=completed_1based,
                    train_t_loss=last_loss_t_value,
                    vis_t_loss=viz_loss_t_render,
                    vis_t_loss2=viz_loss_t_render2,
                )
            train_state.epoch += 1
        if accelerator.is_main_process:
            self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state)
            if self.tb_writer is not None:
                self.tb_writer.flush()
                self.tb_writer.close()
        _append_finetune_train_log(
            reason="completed",
            loss_t_value=last_loss_t_value,
        )
        accelerator.print("Training complete!")

    def save_checkpoint(self, accelerator, cfg, model, optimizer, scheduler, train_state):
        ckpt_file = osp.join(self.exp_dir, f'step{train_state.step:06d}.pth')
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        checkpoint_dict = {
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': train_state.epoch,
            'step': train_state.step,
            'best_val': train_state.best_val,
            'cfg': cfg
        }
        accelerator.save(checkpoint_dict, ckpt_file)
        print(f"ckpt saved to {ckpt_file}")
        # Keep only the latest step checkpoint to save disk space.
        ckpt_files = sorted(glob(osp.join(self.exp_dir, "step*.pth")))
        for old_ckpt in ckpt_files[:-1]:
            if osp.basename(old_ckpt) != osp.basename(ckpt_file):
                try:
                    os.remove(old_ckpt)
                    print(f"removed old ckpt: {old_ckpt}")
                except OSError as e:
                    print(f"warning: failed to remove old ckpt {old_ckpt}: {e}")

    def eval_model(self, cfg, model, train_state, val_dataloader):
        model.eval()
        loss_val = []
        loss_val_r, loss_val_t = [], []
        loss_val_acc = []
        for step_val, batch in enumerate(tqdm(val_dataloader)):
            if step_val >= cfg.max_step_val:
                break
            with torch.no_grad():
                loss, loss_r, loss_t, loss_acc, _ = self.forward_step(batch, cfg, model, vis=step_val == 0)
            loss_val.append(loss.item())
            loss_val_r.append(loss_r.item())
            loss_val_t.append(loss_t.item())
            loss_val_acc.append(loss_acc.item())
        loss_val = np.mean(loss_val)
        if train_state.best_val is None or loss_val < train_state.best_val:
            train_state.best_val = loss_val
        if not cfg.no_wandb and self.accelerator.is_main_process:
            # logging only using the main process
            self._log_metrics(
                {
                    'loss_val': loss_val,
                    'loss_val_r': np.mean(loss_val_r),
                    'loss_val_t': np.mean(loss_val_t),
                    'loss_val_acc': np.mean(loss_val_acc),
                },
                train_state.step,
            )
        elif self.accelerator.is_main_process:
            self._log_metrics(
                {
                    'loss_val': loss_val,
                    'loss_val_r': np.mean(loss_val_r),
                    'loss_val_t': np.mean(loss_val_t),
                    'loss_val_acc': np.mean(loss_val_acc),
                },
                train_state.step,
            )
        if self.accelerator.is_main_process and self.tb_writer is not None:
            self.tb_writer.add_scalar("val/loss", loss_val, train_state.step)
            self.tb_writer.add_scalar("val/loss_r", np.mean(loss_val_r), train_state.step)
            self.tb_writer.add_scalar("val/loss_t", np.mean(loss_val_t), train_state.step)
            self.tb_writer.add_scalar("val/loss_acc", np.mean(loss_val_acc), train_state.step)
        print(f'--- Eval at step {train_state.step}, loss: {loss_val:.4f} lr: {self.optimizer.param_groups[0]["lr"]:.5f} ---')
        self._set_model_mode(True)

    def _compute_logged_t_loss_from_forward(self, batch, cfg, trans_delta_gt, trans_delta_pred, out_dict):
        """Delta-translation L1/MSE mean in normalized space (run_horefine-style)."""
        frame_mask = batch["frame_mask"].unsqueeze(-1)
        bs, t = frame_mask.shape[:2]
        loss_type = str(cfg["loss_type"])
        if "l1" in loss_type:
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).reshape(bs, t, -1)
        else:
            loss_t = ((trans_delta_pred - trans_delta_gt) ** 2).reshape(bs, t, -1)
        return (loss_t * frame_mask).mean()

    def forward_step(self, batch, cfg, model, vis=False):
        "one model forward and return loss"
        # pre-trained model: A is the rendered, B is the input
        rot_delta_pred, rot_delta_gt, trans_delta_gt, trans_delta_pred, out_dict = self.forward_batch(batch, cfg, model, vis, ret_dict=True)
        loss_acc = torch.tensor(0, device=rot_delta_gt.device)
        loss_t_train_delta_masked = None  # masked delta-t from this forward (train mode); see return
        if cfg['loss_type'] == 'l1':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss =  loss_r + loss_t
        elif cfg['loss_type'] == 'l1+self-acc':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            poseA = batch['pose_perturbed']
            B_in_cams, B_in_cams_gt = self.compute_abspose(poseA.shape[0], batch, cfg, poseA, rot_delta_pred, rot_delta_gt,
                                                           trans_delta_gt, trans_delta_pred)
            d1 = geodesic_distance(B_in_cams[:, 1:-1, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :-2, :3, :3].reshape(-1, 3, 3))
            d2 = geodesic_distance(B_in_cams[:, 1:-1, :3, :3].reshape(-1, 3, 3),
                                   B_in_cams[:, 2:, :3, :3].reshape(-1, 3, 3)) # (B, t-2, )
            loss_acc = torch.abs(d1 - d2).mean() * self.cfg.lw_acc
            loss = loss + loss_acc
        elif cfg['loss_type'] == 'l1-abs':
            # predicting absolute pose
            pose_gt = batch['pose_gt'] # (B, T, 4, 4)
            rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1))
            loss_r = torch.abs(rot_delta_pred - rot_gt_axis).mean() * cfg['w_rot']
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            loss_t = torch.abs(trans_delta_pred - trans_gt).mean()
            loss = loss_r + loss_t
            # it is not able to predict depth?
        elif cfg['loss_type'] == 'l1-abs-delta':
            # model predicts both delta and abs pose
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']

            # abs pose error
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            if self.cfg.rot_rep == 'axis_angle':
                rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1)) # BT, 3
                loss_r_abs = torch.abs(out_dict['rot_abs'] - rot_gt_axis).mean() * cfg.w_abs_rot
            elif self.cfg.rot_rep == '6d':
                rot_gt_axis = pose_gt[:, :, :3, 0:3].reshape(-1, 3, 3).view(-1, 6)
                loss_r_abs = torch.abs(out_dict['rot_abs'] - rot_gt_axis).mean() * cfg.w_abs_rot
            else:
                raise NotImplementedError
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            if cfg.loss_abs_trans_rela:
                trans_gt_rela = pose_gt[:, :, :3, 3].clone() - pose_gt[:, 0:1, :3, 3] # relative to 1st frame
                loss_t_abs = torch.abs(out_dict['trans_abs_rela'] - trans_gt_rela).mean() * cfg.w_abs_trans
            else:
                print('loss directly to final GT translation')
                loss_t_abs = torch.abs(out_dict['trans_abs'] - trans_gt).mean() * cfg.w_abs_trans
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs

            if self.accelerator.is_main_process:
                key = 'train' if self.model.training else 'val'
                self._log_metrics(
                    {f'{key}/loss_t_abs': loss_t_abs, f'{key}/loss_r_abs': loss_r_abs},
                    self.train_state.step,
                )
        elif cfg['loss_type'] == 'l2-abs-delta':
            # model predicts both delta and abs pose
            loss_t = ((trans_delta_pred - trans_delta_gt)**2).mean()
            loss_r = ((rot_delta_pred - rot_delta_gt)**2).mean() * cfg['w_rot']

            # abs pose error
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1)) # BT, 3
            loss_r_abs = ((out_dict['rot_abs'] - rot_gt_axis)**2).mean() * cfg.w_abs_rot
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            if cfg.loss_abs_trans_rela:
                trans_gt_rela = pose_gt[:, :, :3, 3].clone() - pose_gt[:, 0:1, :3, 3] # relative to 1st frame
                loss_t_abs = ((out_dict['trans_abs_rela'] - trans_gt_rela)**2).mean() * cfg.w_abs_trans
            else:
                loss_t_abs = ((out_dict['trans_abs'] - trans_gt)**2).mean() * cfg.w_abs_trans
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs

            if self.accelerator.is_main_process:
                key = 'train' if self.model.training else 'val'
                self._log_metrics(
                    {f'{key}/loss_t_abs': loss_t_abs, f'{key}/loss_r_abs': loss_r_abs},
                    self.train_state.step,
                )
        elif cfg['loss_type'] in ['l1-absrot-delta', 'l1-absrot-delta-hum', 'l2-absrot-delta-humabs', 'l1-absrot-delta-humabs', 'l2-absrot-delta-hum']:
            assert cfg.obj_pose_dim_input in [3, 6], 'must encode object rotation only!'
            loss_func = F.l1_loss if 'l1' in cfg['loss_type'] else F.mse_loss
            key = 'train' if self.model.training else 'val'
            loss_dict = {}
            # Keep r/t logging in delta space regardless of training loss formulation.
            loss_r_log, loss_t_log = None, None

            # model predicts both delta and abs pose, abs pose contains rotation only
            frame_mask = batch['frame_mask'].unsqueeze(-1) # (B, T, 1)
            bs, t = frame_mask.shape[:2]
            FIX_LOSS = 0
            if self.cfg.pred_uncertainty:
                # see https://github.com/martius-lab/beta-nll/blob/master/depth_estimation/models/unet_adaptive_bins.py#L189
                uncert_t = F.softplus(out_dict['trans_uncertainty']) + self.cfg.var_epsilon # (BT, 1) eps to protect
                uncert_r = F.softplus(out_dict['rot_uncertainty']) + self.cfg.var_epsilon
                FIX_LOSS = 20 
                loss_t = 0.5 *(loss_func(trans_delta_pred, trans_delta_gt, reduction='none')/uncert_t + uncert_t.log()+ FIX_LOSS) # * cfg['w_transl']
                loss_t = ((loss_t * (uncert_t.detach() ** cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean()* cfg['w_transl']
                # do the same for r
                loss_r = 0.5 * (loss_func(rot_delta_pred, rot_delta_gt, reduction='none')/uncert_r + uncert_r.log()+ FIX_LOSS)
                loss_r = ((loss_r * (uncert_r.detach()**cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * cfg['w_rot']

                # keep track of the classic loss
                with torch.no_grad():
                    loss_t_raw = (loss_func(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs, t,
                                                                                                    -1) * frame_mask).mean() * cfg['w_transl']
                    loss_r_raw = (loss_func(rot_delta_pred, rot_delta_gt, reduction='none').reshape(bs, t,
                                                                                                -1) * frame_mask).mean() * cfg['w_rot']
                    loss_dict[f'{key}/loss_t_raw'] = loss_t_raw
                    loss_dict[f'{key}/loss_r_raw'] = loss_r_raw
                    loss_t_log = loss_t_raw
                    loss_r_log = loss_r_raw

            else:
                if self.cfg.symm_loss:
                    # consider symmetries. 
                    B_in_cams_interm = self.abspose_from_relative(batch, cfg, batch['pose_perturbed'], out_dict['rot'], out_dict['trans']) # (B, T, 4, 4) 
                    pose_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4) 
                    # simply the smallest rotation and translation error from all symmetries
                    loss_r = loss_func(B_in_cams_interm[:, :, None, :3, :3], pose_gt_symm[:, :, :, :3, :3], reduction='none').sum(dim=(-1, -2)).min(-1)[0] # (B, T, N)
                    # mask with frame_mask and weight 
                    loss_r = (loss_r[:, :, None] * frame_mask).mean() * cfg['w_rot']
                    # same for translation
                    loss_t = loss_func(B_in_cams_interm[:, :, None, :3, 3], pose_gt_symm[:, :, :, :3, 3], reduction='none').sum(dim=(-1)).min(-1)[0] # (B, T, N)
                    loss_t = (loss_t[:, :, None] * frame_mask).mean() * cfg['w_transl']
                    # Log object r/t in delta space even when symmetry loss is computed in absolute pose space.
                    loss_t_log = (loss_func(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * cfg['w_transl']
                    loss_r_log = (loss_func(rot_delta_pred, rot_delta_gt, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * cfg['w_rot']
                else: # [HERE]
                    B_in_cams_interm = self.abspose_from_relative(
                        batch, cfg, batch['pose_perturbed'], out_dict['rot'], out_dict['trans']
                    )  # (B, T, 4, 4)
                    pose_gt = batch['pose_gt']  # (B, T, 4, 4)
                    loss_r = loss_func(
                        B_in_cams_interm[:, :, :3, :3], pose_gt[:, :, :3, :3], reduction='none'
                    ).sum(dim=(-1, -2))
                    loss_r = (loss_r[:, :, None] * frame_mask).mean() * cfg['w_rot']

                    t_loss_space = str(getattr(self.cfg, "t_loss_space", "absolute")).strip().lower()
                    if t_loss_space == "delta":
                        loss_t = (loss_func(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * cfg['w_transl']
                    else:
                        if t_loss_space not in ("absolute", ""):
                            if self.accelerator.is_main_process and self.train_state.step == 0:
                                print(f"[warn] unknown t_loss_space={t_loss_space}, fallback to absolute")
                        loss_t = loss_func(B_in_cams_interm[:, :, :3, 3], pose_gt[:, :, :3, 3], reduction='none'
                        ).sum(dim=(-1))
                        loss_t = (loss_t[:, :, None] * frame_mask).mean() * cfg['w_transl']

                    loss_t_log = loss_t
                    loss_r_log = loss_r

                    # Original
                    # loss_t = (loss_func(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean()* cfg['w_transl']
                    # loss_r = (loss_func(rot_delta_pred, rot_delta_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg['w_rot']
                    # loss_t_log = loss_t
                    # loss_r_log = loss_r

            # abs pose error, TODO: also add symmetry loss here 
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            if self.cfg.rot_rep == 'axis_angle':
                rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1))  # BT, 3
                loss_r_abs = (loss_func(out_dict['rot_abs'], rot_gt_axis, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg.w_abs_rot
            elif self.cfg.rot_rep == '6d':
                if self.cfg.symm_loss:
                    pose_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4) 
                    N = pose_gt_symm.shape[2]
                    pose_gt_symm6d = pose_gt_symm[..., :3, 0:2].reshape(-1, N, 6) # BT, N, 6
                    # compute min of all symmetries 
                    loss_r_abs = loss_func(out_dict['rot_abs'][:, None], pose_gt_symm6d, reduction='none').sum(-1).min(-1)[0] # (B, T, N) -> (B, T)
                    loss_r_abs = (loss_r_abs.reshape(bs, t, 1)*frame_mask).mean() * cfg.w_abs_rot
                else:
                    rot_gt_axis = pose_gt[:, :, :3, 0:2].reshape(-1, 3, 2).reshape(-1, 6)
                    loss_r_abs = (loss_func(out_dict['rot_abs'], rot_gt_axis, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg.w_abs_rot
            else:
                raise NotImplementedError

            loss_t_abs = torch.tensor(0, device=rot_delta_gt.device) # abs do not correct translation
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs
            loss_dict.update(**{f'{key}/loss_t_abs': loss_t_abs, f'{key}/loss_r_abs': loss_r_abs})

            # velocity of the abs object pose 

            # compute additional human pose loss
            if self.cfg.nlf_root is not None:
                if cfg.loss_type in ['l1-absrot-delta-hum', 'l2-absrot-delta-hum']:
                    smpl_delta_r = out_dict['hum_pose']
                    smpl_delta_t = out_dict['hum_trans']

                    if self.cfg.rot_rep_hum == '6d':
                        gt_delta_r = batch['delta_smpl_rot'][:, :, :, :, :2].reshape(-1, 24*6)
                        gt_delta_t = batch['delta_smpl_trans'].reshape(-1, 3)
                        if not self.cfg.pred_uncertainty:
                            loss_hum_t = (loss_func(smpl_delta_t, gt_delta_t, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                            loss_hum_r = (loss_func(smpl_delta_r, gt_delta_r, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                        else:
                            uncert_pose = F.softplus(out_dict['hum_pose_uncertainty']).unsqueeze(-1) + self.cfg.var_epsilon# (BT, 24, 1)
                            uncert_smpl_t = F.softplus(out_dict['hum_trans_uncertainty']) + self.cfg.var_epsilon
                            smpl_delta_r = smpl_delta_r.reshape(-1, 24, 6)
                            gt_delta_r = gt_delta_r.reshape(-1, 24, 6)

                            loss_hum_t = 0.5 * (loss_func(smpl_delta_t, gt_delta_t, reduction='none')/uncert_smpl_t + uncert_smpl_t.log()+ FIX_LOSS)
                            loss_hum_t = ((loss_hum_t * (uncert_smpl_t.detach()**self.cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                            loss_hum_r = 0.5 * (loss_func(smpl_delta_r, gt_delta_r, reduction='none')/uncert_pose + uncert_pose.log()+ FIX_LOSS)
                            loss_hum_r = ((loss_hum_r * (uncert_pose.detach()**self.cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                            # keep track of the classic loss
                            with torch.no_grad():
                                loss_hum_t_raw = (loss_func(smpl_delta_t, gt_delta_t, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_t
                                loss_hum_r_raw = (loss_func(smpl_delta_r, gt_delta_r, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_rot
                                loss_dict[f'{key}/loss_hum_r_raw'] = loss_hum_r_raw
                                loss_dict[f'{key}/loss_hum_t_raw'] = loss_hum_t_raw
                        loss_dict[f'{key}/loss_hum_r'] = loss_hum_r
                        loss_dict[f'{key}/loss_hum_t'] = loss_hum_t

                        loss_hum_j = 0. # joints position loss
                        if self.cfg.w_hum_j > 0.:
                            betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = self.smpl_params_from_pred(batch, out_dict)

                            male_mask = batch['is_male'].reshape(-1).bool() # B*L, same shape as pred_smpl_pose
                            assert len(male_mask) == len(pred_smpl_pose)
                            idx_m, idx_f = male_mask.nonzero(as_tuple=True)[0], (~male_mask).nonzero(as_tuple=True)[0]
                            idx_list, jtrs_pr_list = [], []
                            J = 24 
                            if idx_m.numel() > 0:
                                idx_list.append(idx_m)
                                # use male smpl model to get joints 
                                jts_pr_m = self.smpl_male.get_joints(pose72to156(pred_smpl_pose[idx_m]), betas[idx_m], pred_smpl_t[idx_m]) # [:, :J] # take only the first 23 joints without wrists 
                                jtrs_pr_list.append(jts_pr_m)
                            if idx_f.numel() > 0:
                                idx_list.append(idx_f)
                                jts_pr_f = self.smpl_female.get_joints(pose72to156(pred_smpl_pose[idx_f]), betas[idx_f], pred_smpl_t[idx_f]) # [:, :J] # take only the first 23 joints without wrists 
                                jtrs_pr_list.append(jts_pr_f)
                            jts_pr_list = torch.cat(jtrs_pr_list, dim=0)
                            perm = torch.cat(idx_list, dim=0)    # original positions of each sub-batch
                            jts_pr = jts_pr_list[torch.argsort(perm)]        # (B, ...), restored to original order

                            jts_gt = batch['smpl_jtrs_gt'].reshape(-1, jts_pr.shape[-2], 3) # .reshape(-1, J, 3)
                            if not self.cfg.pred_uncertainty:
                                loss_hum_j = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_j
                            else:
                                loss_hum_j = 0.5 * (loss_func(jts_pr, jts_gt, reduction='none') / uncert_pose + uncert_pose.log() + FIX_LOSS)
                                loss_hum_j = ((loss_hum_j * (uncert_pose.detach() ** self.cfg.beta_nll)).reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_j
                                with torch.no_grad():
                                    loss_hum_j_raw = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t,-1) * frame_mask).mean() * self.cfg.w_hum_j
                                    loss_dict[f'{key}/loss_hum_j_raw'] = loss_hum_j_raw

                        loss_hum_b = 0.
                        if self.cfg.w_hum_b > 0:
                            loss_hum_b = (loss_func(betas, batch['betas_gt'].reshape(-1, 10), reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_b
                            loss_dict[f'{key}/loss_hum_b'] = loss_hum_b
                        loss_dict[f'{key}/loss_hum_j'] = loss_hum_j

                        # add velocity loss
                        loss_velo = 0.
                        if self.cfg.w_hum_velo > 0 or self.cfg.w_obj_velo > 0:
                            # Apply separate velocity regularizers for human joints and object translation.
                            jts_pr = jts_pr.reshape(bs, t, -1, 3)
                            jts_gt = batch['smpl_jtrs_gt']
                            loss_velo_hj = torch.tensor(0.0, device=rot_delta_gt.device)
                            loss_velo_ot = torch.tensor(0.0, device=rot_delta_gt.device)
                            if self.cfg.w_hum_velo > 0:
                                velo_pr_hj = jts_pr[:, 1:] - jts_pr[:, :-1]
                                velo_gt_hj = jts_gt[:, 1:] - jts_gt[:, :-1]
                                loss_velo_hj = F.mse_loss(velo_pr_hj, velo_gt_hj, reduction='none').sum(-1).mean()
                            if self.cfg.w_obj_velo > 0:
                                B_in_cams_interm = self.abspose_from_relative(
                                    batch, cfg, batch['pose_perturbed'], out_dict['rot'], out_dict['trans']
                                )
                                velo_pr_ot = B_in_cams_interm[:, 1:, :3, 3] - B_in_cams_interm[:, :-1, :3, 3]
                                velo_gt_ot = pose_gt[:, 1:, :3, 3] - pose_gt[:, :-1, :3, 3]
                                loss_velo_ot = F.mse_loss(velo_pr_ot, velo_gt_ot, reduction='none').sum(-1).mean()
                            loss_hum_velo = loss_velo_hj * self.cfg.w_hum_velo
                            loss_obj_velo = loss_velo_ot * self.cfg.w_obj_velo
                            loss_velo = loss_hum_velo + loss_obj_velo
                            loss_dict[f'{key}/loss_hum_velo'] = loss_hum_velo
                            loss_dict[f'{key}/loss_obj_velo'] = loss_obj_velo
                        # contact prediction
                        loss_contact = 0.
                        if self.cfg.cont_out_dim > 0 and ('contact_dist_gt' in batch) and ('contact' in out_dict):
                            cont_gt = batch['contact_dist_gt'] # (B, T, 52)
                            cont_gt_hands = cont_gt[:, :, [22, 23+15]]
                            cont_pred = out_dict['contact'].reshape(bs, t, -1)
                            if self.cfg.cont_out_type == 'binary':
                                # bce loss 
                                loss_bce = (F.binary_cross_entropy_with_logits(cont_pred, (cont_gt_hands < self.cfg.cont_mask_thres).float(), reduction='none')*frame_mask ).mean() * self.cfg.w_contact
                                loss_dict[f'{key}/loss_contact'] = loss_bce
                                loss_contact = loss_bce
                            elif self.cfg.cont_out_type == 'distance':
                                # mse loss
                                loss_mse = (F.mse_loss(cont_pred, cont_gt_hands, reduction='none')*frame_mask).mean() * self.cfg.w_contact
                                loss_dict[f'{key}/loss_contact'] = loss_mse
                                loss_contact = loss_mse
                        
                        loss += loss_hum_t + loss_hum_r + loss_hum_j + loss_hum_b + loss_velo + loss_contact
                        r_show = loss_r_log if loss_r_log is not None else loss_r
                        t_show = loss_t_log if loss_t_log is not None else loss_t
                        print(f'step {self.train_state.step} hum_r:{loss_hum_r:.3f}, hum_t:{loss_hum_t:.3f}, hum_j: {loss_hum_j:.3f}, hum_b: {loss_hum_b:.3f}, velo: {loss_velo:.3f}, r_abs:{loss_r_abs:.3f}, t_abs:{loss_t_abs:.3f}, r: {r_show:.3f}, t: {t_show:.3f}, contact: {loss_contact:.3f}, tot: {loss:.3f}')
                    else:
                        raise NotImplementedError
                elif cfg.loss_type in ['l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
                    assert self.cfg.w_hum_j > 0.
                    # now predicting abs
                    trans_pr = out_dict['body_transl'] # (BT, 3)
                    trans_gt = batch['smpl_transl_gt'].reshape(-1, 3)
                    # compute loss on parameters, joints
                    loss_hum_t = (loss_func(trans_pr, trans_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                    rotmat_gt = batch['smpl_rotmat_gt'].reshape(-1, 24, 3, 3)
                    rotmat_pr = out_dict['body_rotmat'] # (BT, 24, 3, 3)
                    loss_hum_r = (loss_func(rotmat_pr, rotmat_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                    bs = batch['smpl_transl_gt'].shape[0]
                    betas = batch['betas_gt'].reshape(self.cfg.clip_len * bs, 10)
                    jts_pr = self.smpl_model.get_joints(rotmat_pr.reshape(self.cfg.clip_len * bs, 24*9), betas, trans_pr, axis2rot=False)  # (BT, J, 3)
                    jts_gt = batch['smpl_jtrs_gt'].reshape(-1, 24, 3)
                    loss_hum_j = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean().item() * self.cfg.w_hum_j
                    loss += loss_hum_t + loss_hum_r + loss_hum_j

                    loss_dict[f'{key}/loss_hum_r'] = loss_hum_r
                    loss_dict[f'{key}/loss_hum_t'] = loss_hum_t
                    loss_dict[f'{key}/loss_hum_j'] = loss_hum_j


            if self.accelerator.is_main_process:
                self._log_metrics(loss_dict, self.train_state.step)

        elif cfg['loss_type'] == 'l1+geo-acc':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            poseA = batch['pose_perturbed']
            B_in_cams, B_in_cams_gt = self.compute_abspose(poseA.shape[0], batch, cfg, poseA, rot_delta_pred, rot_delta_gt,
                                                           trans_delta_gt, trans_delta_pred)
            d1 = geodesic_distance(B_in_cams[:, 1:, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :-1, :3, :3].reshape(-1, 3, 3))
            d2 = geodesic_distance(B_in_cams_gt[:, 1:, :3, :3].reshape(-1, 3, 3),
                                   B_in_cams_gt[:, :-1, :3, :3].reshape(-1, 3, 3))
            loss_acc = torch.abs(d1 - d2).mean() * self.cfg.lw_acc
            print(f'loss_acc: {loss_acc:.4f}, loss: {loss:.4f}')
            loss = loss + loss_acc

        elif cfg['loss_type'] == 'l2': # default L2
            loss_t = ((trans_delta_pred - trans_delta_gt) ** 2).mean()
            loss_r = ((rot_delta_pred - rot_delta_gt) ** 2).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            # add acceleration loss
            if self.cfg.lw_acc>0:
                trans_delta_uno = trans_delta_pred * batch['mesh_diameter'].reshape(len(trans_delta_pred), -1) / 2.
                rot_delta_uno = so3_exp_map(rot_delta_pred * self.cfg['rot_normalizer']).permute(0, 2, 1)
                # now convert to (B, T...)
                poseA = batch['pose_perturbed'] # (B, T, 4, 4)
                poseB = batch['pose_gt']
                B, T = poseA.shape[:2]
                pose_corrected = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4), trans_delta=trans_delta_uno,
                                                          rot_mat_delta=rot_delta_uno)
                pose_corrected = pose_corrected.reshape(B, T, 4, 4)

                acc_t_gt = poseB[:, :-2, :3, 3] - 2 * poseB[:, 1:-1, :3, 3] + poseB[:, 2:, :3, 3]
                acc_t_pred = pose_corrected[:, :-2, :3, 3] - 2 * pose_corrected[:, 1:-1, :3, 3] + pose_corrected[:, 2:, :3, 3]

                axis_gt = so3_log_map(poseB[:, :, :3, :3].reshape(-1, 3, 3)).reshape(B, T, -1)
                axis_pr = so3_log_map(pose_corrected[:, :, :3, :3].reshape(-1, 3, 3)).reshape(B, T, -1)
                acc_r_gt = axis_gt[:, :-2] - 2 * axis_gt[:, 1:-1] + axis_gt[:, 2:]
                acc_r_pr = axis_pr[:, :-2] - 2 * axis_pr[:, 1:-1] + axis_pr[:, 2:]
                la_t = F.l1_loss(acc_t_pred, acc_t_gt).mean()
                la_r = F.l1_loss(acc_r_gt, acc_r_pr).mean()
                loss_acc = (la_r + la_t) * self.cfg.lw_acc
            loss = loss + loss_acc

            # this needs to be computed in the original pose space.
        else:
            raise RuntimeError

        # For training logs/TensorBoard, expose delta-space object r/t when available.
        if cfg['loss_type'] in ['l1-absrot-delta', 'l1-absrot-delta-hum', 'l2-absrot-delta-humabs', 'l1-absrot-delta-humabs', 'l2-absrot-delta-hum']:
            if loss_r_log is not None:
                loss_r = loss_r_log
            if loss_t_log is not None:
                loss_t = loss_t_log

        # Same forward as loss_t (train mode): masked delta t metric — avoids a second forward in eval().
        if cfg['loss_type'] in ['l1-absrot-delta', 'l1-absrot-delta-hum', 'l2-absrot-delta-humabs', 'l1-absrot-delta-humabs', 'l2-absrot-delta-hum']:
            frame_mask_td = batch['frame_mask'].unsqueeze(-1)
            bs_td, t_td = frame_mask_td.shape[:2]
            loss_func_td = F.l1_loss if 'l1' in cfg['loss_type'] else F.mse_loss
            loss_t_train_delta_masked = (
                loss_func_td(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs_td, t_td, -1)
                * frame_mask_td
            ).mean() * cfg['w_transl']
        elif cfg['loss_type'] in ['l1', 'l1+self-acc']:
            frame_mask_td = batch['frame_mask'].unsqueeze(-1)
            bs_td, t_td = frame_mask_td.shape[:2]
            loss_t_train_delta_masked = (
                torch.abs(trans_delta_pred - trans_delta_gt).reshape(bs_td, t_td, -1) * frame_mask_td
            ).mean()

        return loss, loss_r, loss_t, loss_acc, loss_t_train_delta_masked

    def forward_batch(self, batch, cfg, model, vis=False, ret_dict=False):
        "forward one batch"
        imgsB, imgsA = batch['input_rgbs'], batch['render_rgbs']
        xyzB, xyzA = batch['input_xyz'], batch['render_xyz']
        pose_perturbed = self._get_poseA_norm(batch)
        output = model(torch.cat([imgsA, xyzA], 2), torch.cat([imgsB, xyzB], 2), pose_perturbed, batch)
        # Never mutate batch GT tensors in-place here; downstream fingerprint checks rely on stable GT values.
        delta_transl_raw, delta_rot_raw = self._get_obj_delta_supervision(batch)
        trans_delta_gt = delta_transl_raw.clone()  # (B, T, 3)
        mesh_radius = batch['mesh_diameter'] / 2.  # (B, T)
        trans_normalizer = batch['trans_normalizer']  # (B, T, 3)
        B, T = trans_delta_gt.shape[:2]
        # use diameter: the xyz map is normalized by object diameter
        if cfg['normalize_xyz']:
            trans_delta_gt *= 1 / mesh_radius.reshape(len(trans_delta_gt), T, -1)
        else:
            trans_delta_gt = trans_delta_gt / trans_normalizer
            if not (torch.abs(trans_delta_gt) <= 1 + 1e-3).all():
                logging.info("ERROR label")
        rot_delta_mat_gt = delta_rot_raw
        rot_delta_gt = so3_log_map(rot_delta_mat_gt.reshape(B * T, 3, 3).permute(0, 2, 1))  # permute: pyt3d so3 uses col order
        rot_delta_gt = rot_delta_gt / cfg['rot_normalizer']  # random noise sample range.

        # from scipy.spatial.transform import Rotation as R
        # import numpy as np
        # gt = batch['smpl_poses_gt'][0].detach().cpu().numpy()  # (T, 156)
        # gt_aa72 = gt[:, :72]
        # inp_rm = batch['nlf_rotmat'].detach().cpu().numpy()
        # # nlf_rotmat can be either (B, T, J, 3, 3) or (BT, J, 3, 3).
        # if inp_rm.ndim == 5:
        #     inp_rm_t = inp_rm[0]  # (T, J, 3, 3)
        # elif inp_rm.ndim == 4:
        #     inp_rm_t = inp_rm  # (BT, J, 3, 3)
        # else:
        #     raise RuntimeError(f"unexpected nlf_rotmat shape: {inp_rm.shape}")
        # t_cmp = min(gt_aa72.shape[0], inp_rm_t.shape[0])
        # j_cmp = min(24, inp_rm_t.shape[1])
        # inp_aa = R.from_matrix(inp_rm_t[:t_cmp, :j_cmp].reshape(-1, 3, 3)).as_rotvec().reshape(t_cmp, j_cmp, 3)
        # inp_aa72 = inp_aa.reshape(t_cmp, j_cmp * 3)
        # gt_cmp = gt_aa72[:t_cmp, : j_cmp * 3]
        # pose_mae = float(np.mean(np.abs(gt_cmp - inp_aa72)))
        # pose_max = float(np.max(np.abs(gt_cmp - inp_aa72)))
        # print(f"[debug] gt-vs-input pose mae={pose_mae:.6f}, max={pose_max:.6f}, T={t_cmp}, J={j_cmp}")
        # import pdb; pdb.set_trace()
        
        # WARNING: Debug-only ablation.
        # This replaces model-predicted object delta rotation with GT immediately after CoCONet output.
        # Keep disabled in normal training/eval (`debug_force_gt_obj_rot=False`).
        if getattr(cfg, "debug_force_gt_obj_rot", False):
            with torch.no_grad():
                pred_rot_mae = torch.abs(output['rot'].float() - rot_delta_gt).mean()
                pred_trans_mae = torch.abs(output['trans'].float() - trans_delta_gt.reshape(B * T, 3)).mean()
            key = 'train' if model.training else 'val'
            self._log_metrics(
                {
                    f'{key}/debug_rot_mae_before_gt_inject': pred_rot_mae,
                    f'{key}/debug_trans_mae_before_gt_inject': pred_trans_mae,
                },
                self.train_state.step,
            )
            output['rot'] = rot_delta_gt.detach().clone()

        trans = output['trans'].float()  # (BT,3)
        rot = output['rot'].float()  # BT, 3
        trans_delta_pred = trans
        trans_delta_gt = trans_delta_gt.reshape(B * T, 3)  # FP was trained to predict

        # log error
        if self.accelerator.is_main_process:
            with torch.no_grad():
                log_dict = {}
                poseA = batch['pose_perturbed']
                B_in_cams, B_in_cams_gt = self.compute_abspose(B, batch, cfg, poseA, rot, rot_delta_gt,
                                                               trans_delta_gt, trans_delta_pred, output)
                err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams[:, :, :3, 3]) ** 2, -1).sqrt().mean()
                err_r = geodesic_distance(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :, :3, :3].reshape(-1, 3, 3)).mean()
                key = 'train' if model.training else 'val'
                log_dict[f'{key}/err_t'] = err_t
                log_dict[f'{key}/err_r'] = err_r
                log_dict[f'{key}/err_r_deg'] = err_r * 180/torch.pi

                # log contact accuracy 
                if self.cfg.cont_out_dim > 0 and ('contact_dist_gt' in batch) and ('contact' in output):
                    cont_gt = batch['contact_dist_gt'] # (B, T, 52)
                    cont_gt_hands = cont_gt[:, :, [22, 23+15]]
                    cont_pred = output['contact'].reshape(B, T, -1)
                    if self.cfg.cont_out_type == 'binary':
                        cont_acc = (cont_pred > 0).float() == (cont_gt_hands < self.cfg.cont_mask_thres).float()
                    elif self.cfg.cont_out_type == 'distance':
                        cont_acc = (cont_pred < self.cfg.cont_mask_thres).float() == (cont_gt_hands < self.cfg.cont_mask_thres).float()
                    log_dict[f'{key}/cont_acc'] = cont_acc.float().mean()
                # log error of intermediate predictions
                if self.cfg['loss_type'] in ['l1-abs-delta', 'l1-absrot-delta', 'l1-absrot-delta-hum', 'l1-absrot-delta-humabs', 'l2-absrot-delta-humabs']:
                    B_in_cams_interm = self.abspose_from_relative(batch, cfg, poseA, output['rot'], output['trans'])
                    # also compute symmetries 
                    if self.cfg.symm_loss and ('pose_gt_symm' in batch):
                        B_in_cams_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4)
                        err_r = geodesic_distance(B_in_cams_gt_symm[:, :, :, :3, :3].reshape(-1, 3, 3),
                                              B_in_cams_interm[:, :, None, :3, :3].repeat(1, 1, B_in_cams_gt_symm.shape[2], 1, 1).reshape(-1, 3, 3)).reshape(-1, B_in_cams_gt_symm.shape[2]).min(-1)[0]
                        err_r = err_r.mean()
                        # do the same for translation
                        err_t = torch.sum((B_in_cams_gt_symm[:, :, :, :3, 3].reshape(-1, 3) - B_in_cams_interm[:, :, None, :3, 3].repeat(1, 1, B_in_cams_gt_symm.shape[2], 1).reshape(-1, 3)) ** 2, -1).sqrt()
                        err_t = err_t.reshape(-1, B_in_cams_gt_symm.shape[2]).min(-1)[0].mean()
                    else:
                        err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams_interm[:, :, :3, 3]) ** 2, -1).sqrt().mean()
                        err_r = geodesic_distance(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3),
                                                B_in_cams_interm[:, :, :3, :3].reshape(-1, 3, 3)).mean()

                    key = 'train' if model.training else 'val'
                    log_dict[f'{key}/err_interm_t'] = err_t
                    log_dict[f'{key}/err_interm_r'] = err_r
                self._log_metrics(log_dict, self.train_state.step)

        # visualize input and output predictions
        if vis and self.accelerator.is_main_process:
            start = time.time()
            bid = 0  # batch id
            skip = 16  # log every N frame

            key = 'train' if self.model.training else 'val'
            log_dict = {}
            maskA, maskB, rgbsA, rgbsB, xyzA, xyzB = self.prepare_input_viz(batch, cfg)
            poseB = batch['pose_gt'] # this does not match B_in_cams_gt!
            # TODO: replace poseB with poseA + delta GT

            poseA = batch['pose_perturbed']
            to_origin = batch['to_origin'][bid].cpu().numpy()
            bbox = batch['obj_bbox_3d'][bid].cpu().numpy()
            clip_len = rgbsA.shape[1]

            # log error
            with torch.no_grad():
                B_in_cams, B_in_cams_gt = self.compute_abspose(B, batch, cfg, poseA, rot, rot_delta_gt,
                                                               trans_delta_gt, trans_delta_pred, output)
                err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams[:, :, :3, 3])**2, -1).sqrt().mean()
                err_r = geodesic_distance(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3), B_in_cams[:, :, :3, :3].reshape(-1, 3, 3)).mean()
                key = 'train' if model.training else 'val'
                log_dict[f'{key}/err_t'] = err_t
                log_dict[f'{key}/err_r'] = err_r

                # log SMPL evaluation
                NJ = 24
                if cfg.nlf_root is not None:
                    verts_smpl_gt, verts_smpl, jtrs_gt, jtrs_pr, pred_smpl_r, pred_smpl_t, jts_rot_pr, jts_rot_gt = self.compute_smpl_verts(
                        batch, output)
                    v2v = torch.sum((verts_smpl_gt - verts_smpl) ** 2, -1).sqrt().mean()
                    mpjpe = torch.sum((jtrs_pr - jtrs_gt) ** 2, -1).sqrt().mean()
                    mpjae = Utils.geodesic_distance_batch(jts_rot_pr, jts_rot_gt).mean()
                    ste = torch.sum((batch['smpl_transl_gt'].reshape(-1, 3) - pred_smpl_t) ** 2).sqrt().mean()
                    log_dict[f'{key}/v2v'] = v2v
                    log_dict[f'{key}/mpjpe'] = mpjpe
                    log_dict[f'{key}/mpjae'] = mpjae
                    log_dict[f'{key}/smpl_t'] = ste

            for i in range(0, clip_len, skip):
                comb, rgba, rgbb = self.visualize_rgbm(batch, bid, i, maskA, maskB, rgbsA, rgbsB)
                # add xyz as well
                xyza_vis = (np.clip(xyzA[bid, i].transpose(1, 2, 0)+0.5, 0, 1.)* 255).astype(np.uint8)
                xyzb_vis = (np.clip(xyzB[bid, i].transpose(1, 2, 0)+0.5, 0, 1.)* 255).astype(np.uint8)
                comb = np.concatenate([comb, np.concatenate([xyza_vis, xyzb_vis], 0)], axis=1)
                # Visualize pose predictions as well
                K = batch['K_rois'][bid, i].cpu().numpy()
                center_pose = B_in_cams_gt[bid, i].cpu().numpy() @ np.linalg.inv(to_origin)
                vis_gt, vis_input, vis_pred = rgbb.copy(), rgba.copy(), rgbb.copy()
                vis_gt = Utils.draw_posed_3d_box(K, img=vis_gt, ob_in_cam=center_pose, bbox=bbox, line_color=(0, 255, 0))
                vis_gt = Utils.draw_xyz_axis(vis_gt, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                             transparency=0, is_input_rgb=True)
                center_pose = poseA[0, i].cpu().numpy() @ np.linalg.inv(to_origin)
                vis_input = Utils.draw_posed_3d_box(K, img=vis_input, ob_in_cam=center_pose, bbox=bbox, line_color=(255, 0, 0))
                vis_input = Utils.draw_xyz_axis(vis_input, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                             transparency=0, is_input_rgb=True)

                pose = B_in_cams[bid, i].detach().cpu().numpy()
                center_pose = pose @ np.linalg.inv(to_origin)
                vis_pred = Utils.draw_posed_3d_box(K, img=vis_pred, ob_in_cam=center_pose, bbox=bbox, line_color=(0, 255, 255))
                vis_pred = Utils.draw_xyz_axis(vis_pred, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                                transparency=0, is_input_rgb=True)

                # now show an overlap
                vis_comb = vis_gt.copy()
                vis_comb = Utils.draw_posed_3d_box(K, img=vis_comb, ob_in_cam=center_pose, bbox=bbox,
                                                   line_color=(0, 255, 255))
                vis_comb = Utils.draw_xyz_axis(vis_comb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                               transparency=0, is_input_rgb=True)

                # add contact text 
                if self.cfg.cont_out_dim > 0 and ('contact_dist_gt' in batch) and ('contact' in output):
                    cont_gt = batch['contact_dist_gt'][bid, i, [22, 23+15]]
                    cont_text = f'lh: {cont_gt[0]:.3f}, rh: {cont_gt[1]:.3f}'
                    cv2.putText(vis_gt, cont_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                    # add to vis_pred as well 
                    cont_pred = output['contact'].reshape(B, T, -1)[bid, i]
                    cont_text = f'lh: {cont_pred[0]:.3f}, rh: {cont_pred[1]:.3f}'
                    cv2.putText(vis_pred, cont_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                pose_comb = np.concatenate([np.concatenate([vis_input, vis_pred], 1),
                                            np.concatenate([vis_gt, vis_comb], 1)], axis=0)
                comb = np.concatenate([comb, pose_comb], axis=1)

                # visualize PC
                pc_ab, pc_colors = self.visualize_xyz_map(bid, i, xyzA, xyzB)
                log_dict[f'{key}_xyz_{i}'] = wandb.Object3D(np.concatenate([pc_ab, pc_colors], 1),
                                                            caption=f'xyz: red-A green-B ')

                log_dict[f'{key}_all_{i}'] = wandb.Image(comb, caption='top-A bottom-B')
                outfile = osp.join(self.exp_dir, f'vis/{key}_xyz_{i}.ply')
                os.makedirs(osp.dirname(outfile), exist_ok=True)
                trimesh.PointCloud(pc_ab, colors=pc_colors).export(outfile)
                outfile = osp.join(self.exp_dir, f'vis/{key}_all_{i}.png')
                Image.fromarray(comb).save(outfile)

            if self.accelerator.is_main_process:
                self._log_metrics(log_dict, self.train_state.step)
            end = time.time()
            print(f'Step {self.train_state.step} {key} vis uploading finished after {end - start} seconds')

        if ret_dict:
            return rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, output
        return rot, rot_delta_gt, trans_delta_gt, trans_delta_pred

    def prepare_input_viz(self, batch, cfg):
        rgbsA = (batch['render_rgbs'].cpu().numpy() * 255).astype(np.uint8)  # (B, T, 3, H, W)
        rgbsB = (batch['input_rgbs'].cpu().numpy() * 255).astype(np.uint8)  # (B, T, 3, H, W)
        xyzA = (batch['render_xyz'][:, :, :3].cpu().numpy())
        xyzB = (batch['input_xyz'][:, :, :3].cpu().numpy())
        if batch['input_xyz'].shape[2] in [5, 6]:
            maskA = (batch['render_xyz'][:, :, 3:].cpu().numpy() * 255).astype(np.uint8)
            maskB = (batch['input_xyz'][:, :, 3:].cpu().numpy() * 255).astype(np.uint8)
        elif cfg.mask_encode_type == 'one-channel':
            maskA = ((batch['render_xyz'][:, :, 3:].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            maskB = ((batch['input_xyz'][:, :, 3:].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        else:
            maskA, maskB = None, None
        return maskA, maskB, rgbsA, rgbsB, xyzA, xyzB

    def compute_abspose(self, B, batch, cfg, poseA, rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, out_dict=None):
        b, t = poseA.shape[:2]
        if self.cfg['loss_type'] == 'l1-abs':
            rot_pred = so3_exp_map(rot).permute(0, 2, 1)
            trans_pred = trans_delta_pred
            B_in_cams = torch.zeros_like(poseA)
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
            B_in_cams[:, :, :3, 3] = trans_pred.reshape(b, t, 3)
        elif self.cfg['loss_type'] in ['l1-abs-delta', 'l2-abs-delta']:
            if self.cfg.rot_rep == 'axis_angle':
                rot_pred = so3_exp_map(out_dict['rot_abs']).permute(0, 2, 1)
            elif self.cfg.rot_rep == '6d':
                rot_pred = geom_utils.rot6d_to_rotmat(out_dict['rot_abs'])
            else:
                raise NotImplementedError
            trans_pred = out_dict['trans_abs']
            B_in_cams = torch.zeros_like(poseA)
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
            B_in_cams[:, :, :3, 3] = trans_pred.reshape(b, t, 3)
        elif cfg['loss_type'] in ['l1-absrot-delta', 'l1-absrot-delta-hum', 'l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
            # rot from abs, trans from delta
            B_in_cams = self.abspose_from_relative(batch, cfg, poseA, rot, trans_delta_pred)
            if self.cfg.rot_rep == 'axis_angle':
                rot_pred = so3_exp_map(out_dict['rot_abs']).permute(0, 2, 1)
            elif self.cfg.rot_rep == '6d':
                rot_pred = geom_utils.rot6d_to_rotmat(out_dict['rot_abs'])
            else:
                raise NotImplementedError
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
        else:
            B_in_cams = self.abspose_from_relative(batch, cfg, poseA, rot, trans_delta_pred)
        rot_delta_gt_rot = so3_exp_map(rot_delta_gt * cfg['rot_normalizer']).permute(0, 2, 1)
        B_in_cams_gt = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4),
                                                           trans_delta=trans_delta_gt * batch['mesh_diameter'].reshape((-1, 1)) / 2.,
                                                           rot_mat_delta=rot_delta_gt_rot).reshape(B, t, 4, 4)  # (BT, 4, 4)

        return B_in_cams, B_in_cams_gt

    def abspose_from_relative(self, batch, cfg, poseA, rot, trans_delta_pred):
        "compute abs pose from relative pose prediction"
        b, t = poseA.shape[:2]
        trans_delta_final = trans_delta_pred * batch['mesh_diameter'].reshape((-1, 1)) / 2.  # undo normalization
        rot_delta_final = so3_exp_map(rot * cfg['rot_normalizer']).permute(0, 2, 1)
        B_in_cams = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4), trans_delta=trans_delta_final,
                                                        rot_mat_delta=rot_delta_final).reshape(b, t, 4, 4)
        return B_in_cams

    def compute_smpl_verts(self, batch, out_dict):
        "compute smpl verts for GT and prediction"
        betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = self.smpl_params_from_pred(batch, out_dict)

        verts_smpl, jtrs_pr, _, _, jts_rot_pr = self.smpl_male(pose72to156(pred_smpl_pose), betas, pred_smpl_t, ret_glb_rot=True)
        verts_smpl_gt, jtrs_gt, _, _, jts_rot_gt = self.smpl_male(batch['smpl_poses_gt'].reshape(-1, 156), betas,
                                                                   batch['smpl_transl_gt'].reshape(-1, 3), ret_glb_rot=True)
        return verts_smpl_gt, verts_smpl, jtrs_gt, jtrs_pr, pred_smpl_r, pred_smpl_t, jts_rot_pr, jts_rot_gt

    def smpl_params_from_pred(self, batch, out_dict):
        "compute SMPL parameters from prediction, return in shape (BT, ...)"
        J = 24
        if self.cfg.loss_type in ['l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
            # predict abs pose already
            pred_smpl_r = out_dict['body_rotmat']
            pred_smpl_t = out_dict['body_transl']
        else:
            # additional visualization for human as well
            # Prefer hum_*_init keys (video_data_vis / run_horefine path),
            # keep nlf_* fallback for legacy training datasets.
            if 'hum_pose_init' in batch:
                hum_pose_init = batch['hum_pose_init'].reshape(-1, 52, 3)[:, :J]  # (BT, 24, 3)
                nlf_poses = so3_exp_map(hum_pose_init.reshape(-1, 3)).reshape(-1, J, 3, 3)
            else:
                nlf_poses = batch['nlf_rotmat'].reshape(-1, J, 3, 3)  # B, T, J, 3, 3

            if 'hum_transl_init' in batch:
                pred_smpl_t = batch['hum_transl_init'].reshape(-1, 3)
            else:
                pred_smpl_t = batch['nlf_transl'].reshape(-1, 3) # + out_dict['hum_trans']
            delta_pr_r = geom_utils.rot6d_to_rotmat(out_dict['hum_pose'].reshape(-1, 6)).reshape(-1, J, 3, 3)

            pred_smpl_r = delta_pr_r @ nlf_poses
        pred_smpl_pose = geom_utils.rotation_matrix_to_angle_axis(pred_smpl_r.reshape(-1, 3, 3)).reshape(-1, J * 3)
        betas_base = batch['hum_betas_init'] if 'hum_betas_init' in batch else batch['betas_nlf']
        if 'hum_shape' in out_dict:
            betas = out_dict['hum_shape'] + betas_base.reshape(-1, 10)
        else:
            betas = betas_base.reshape(-1, 10) # use initial human betas
        return betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t

    @staticmethod
    def visualize_rgbm(batch, bid, i, maskA, maskB, rgbsA, rgbsB):
        rgba = rgbsA[bid, i].transpose(1, 2, 0)
        rgbb = rgbsB[bid, i].transpose(1, 2, 0)
        ab = np.concatenate([rgba, rgbb], axis=0)
        # log mask as well
        if batch['input_xyz'].shape[2] == 5:
            maska = maskA[bid, i].transpose(1, 2, 0)  # already (H, W, 2)
            maskb = maskB[bid, i].transpose(1, 2, 0)
            comb = np.concatenate([np.concatenate([maska, np.zeros_like(maska[:, :, 0:1])], -1),
                                   np.concatenate([maskb, np.zeros_like(maskb[:, :, 0:1])], -1)], 0)
            comb = np.concatenate([ab, comb], axis=1)
        elif batch['input_xyz'].shape[2] == 4:
            # one channel
            maska_h = maskA[bid, i, 0][:, :, None].repeat(3, -1)
            maska_o = maskA[bid, i, 0][:, :, None].repeat(3, -1)
            maskb_h = maskB[bid, i, 0][:, :, None].repeat(3, -1)
            maskb_o = maskB[bid, i, 0][:, :, None].repeat(3, -1)
            comb = np.concatenate([maska_h, maskb_h], axis=0)
            comb = np.concatenate([ab, comb], axis=1)
        elif batch['input_xyz'].shape[2] == 6:
            # do nothing
            maska = maskA[bid, i].transpose(1, 2, 0) # already (H, W, 3)
            maskb = maskB[bid, i].transpose(1, 2, 0)
            comb = np.concatenate([maska, maskb], axis=0)
            comb = np.concatenate([ab, comb], axis=1)
        else:
            comb = ab
        return comb, rgba, rgbb

    @staticmethod
    def visualize_xyz_map(bid, i, xyzA, xyzB):
        mask_xyza = np.abs(xyzA[bid, i, 2]) > 0.001  # avoid all zeros
        pc_a = xyzA[bid, i].transpose(1, 2, 0)[mask_xyza].reshape((-1, 3))
        mask_xyzb = np.abs(xyzB[bid, i, 2]) > 0.001
        pc_b = xyzB[bid, i].transpose(1, 2, 0)[mask_xyzb].reshape((-1, 3))
        pc_ab = np.concatenate([pc_a, pc_b], axis=0)
        red = np.array([[255, 0, 0]]).repeat(len(pc_a), 0)
        green = np.array([[0, 255, 0]]).repeat(len(pc_b), 0)
        pc_colors = np.concatenate([red, green], 0)
        return pc_ab, pc_colors

def main2():
    # 1. Create the base config from the structured dataclass
    # This holds all the defaults.
    cfg = get_config()

    trainer = Trainer(cfg)
    trainer.train()


def get_config():
    base_conf = OmegaConf.structured(TrainTemporalRefinerConfig)
    cfg_cli = OmegaConf.from_cli()
    # 2. Load the config from the YAML file
    # This holds our overrides.
    if 'config' in cfg_cli:
        file_conf = OmegaConf.load(cfg_cli.config)
        # 3. Merge the two configurations.
        # The values in `file_conf` will overwrite the defaults in `base_conf`.
        cfg: TrainTemporalRefinerConfig = OmegaConf.merge(base_conf, file_conf)
        print("Overriding config from file", cfg_cli.config)
    else:
        cfg = base_conf
    # merge with command line args
    cfg = OmegaConf.merge(cfg, cfg_cli)
    return cfg


if __name__ == "__main__":
    main2()

