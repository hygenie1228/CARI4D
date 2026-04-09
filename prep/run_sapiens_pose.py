# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Run Sapiens 2D pose estimation (COCO 17 keypoints) on video frames and save
results in the same packed pkl format expected by the CARI4D pipeline.

Output: <packed_root>/<seq_name>_GT-packed.pkl
  - 'frames': list of 6-digit zero-padded frame indices
  - 'joints2d': ndarray (N, 1, 17, 3) float64, last dim = (x, y, confidence)

Usage:
  python prep/run_sapiens_pose.py --video <path> --masks_root <path> --packed_root <path> \
      [--sapiens_root sapiens] [--checkpoint <path>] [--device cuda:0]
"""

import sys, os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import h5py
import cv2
import joblib
from tqdm import tqdm
import os.path as osp

# Add sapiens pose to path
SAPIENS_POSE_ROOT = osp.join(os.getcwd(), 'sapiens', 'pose')
sys.path.insert(0, SAPIENS_POSE_ROOT)

# Use sapiens' own mmpretrain (has sapiens arch definitions) instead of pip-installed one
SAPIENS_PRETRAIN_ROOT = osp.join(os.getcwd(), 'sapiens', 'pretrain')
sys.path.insert(0, SAPIENS_PRETRAIN_ROOT)
# Remove pip-installed mmpretrain from module cache so sapiens' version is used
for _k in list(sys.modules.keys()):
    if _k.startswith('mmpretrain'):
        del sys.modules[_k]

# Patch transformers compatibility for mmpretrain (functions moved in newer versions)
import transformers.modeling_utils as _tmu
for _name in ('apply_chunking_to_forward', 'find_pruneable_heads_and_indices', 'prune_linear_layer'):
    if not hasattr(_tmu, _name):
        import transformers.pytorch_utils as _tpu
        setattr(_tmu, _name, getattr(_tpu, _name))
for _name in ('GenerationMixin',):
    if not hasattr(_tmu, _name):
        try:
            from transformers import generation_utils as _gu
            setattr(_tmu, _name, getattr(_gu, _name, type('_Stub', (), {})))
        except Exception:
            setattr(_tmu, _name, type('_Stub', (), {}))
if not hasattr(_tmu, 'GenerationConfig'):
    from transformers import GenerationConfig
    _tmu.GenerationConfig = GenerationConfig

import mmpretrain.models.backbones  # noqa: registers sapiens VisionTransformer

# Patch torch.load for PyTorch 2.6+ (weights_only=True by default breaks old checkpoints)
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def get_person_bbox_from_mask(mask, pad_ratio=0.1):
    """Get bounding box from binary person mask with padding."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w, h = x2 - x1, y2 - y1
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(mask.shape[1], x2 + pad_x)
    y2 = min(mask.shape[0], y2 + pad_y)
    return np.array([x1, y1, x2, y2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--masks_root', required=True, help='Root dir of mask H5 files')
    parser.add_argument('--packed_root', required=True, help='Output root for packed pkl')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to Sapiens checkpoint. If not given, auto-detect.')
    parser.add_argument('--config', default=None,
                        help='Path to Sapiens config. If not given, auto-detect.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference (reduce if OOM)')
    args = parser.parse_args()

    seq_name = osp.basename(args.video).split('.')[0]

    # auto-detect checkpoint
    sapiens_ckpt_root = osp.expanduser('~/sapiens_host/pose/checkpoints')
    if args.checkpoint is None:
        # try 0.3b first, then 0.6b
        for model_size, ckpt_name in [
            ('sapiens_0.3b', 'sapiens_0.3b_coco_best_coco_AP_796.pth'),
            ('sapiens_0.6b', 'sapiens_0.6b_coco_best_coco_AP_812.pth'),
        ]:
            p = osp.join(sapiens_ckpt_root, model_size, ckpt_name)
            if osp.isfile(p):
                args.checkpoint = p
                break
        assert args.checkpoint is not None, \
            f'No Sapiens checkpoint found under {sapiens_ckpt_root}. Please download one.'
    print(f'Using Sapiens checkpoint: {args.checkpoint}')

    # auto-detect config
    if args.config is None:
        model_size = 'sapiens_0.3b' if '0.3b' in args.checkpoint else \
                     'sapiens_0.6b' if '0.6b' in args.checkpoint else \
                     'sapiens_1b' if '1b' in args.checkpoint else 'sapiens_2b'
        cfg_name = {
            'sapiens_0.3b': 'sapiens_0.3b-210e_coco-1024x768.py',
            'sapiens_0.6b': 'sapiens_0.6b-210e_coco-1024x768.py',
            'sapiens_1b': 'sapiens_1b-210e_coco-1024x768.py',
            'sapiens_2b': 'sapiens_2b-210e_coco-1024x768.py',
        }[model_size]
        args.config = osp.join(SAPIENS_POSE_ROOT, 'configs', 'sapiens_pose', 'coco', cfg_name)
    print(f'Using Sapiens config: {args.config}')

    # output
    os.makedirs(args.packed_root, exist_ok=True)
    outfile = osp.join(args.packed_root, f'{seq_name}_GT-packed.pkl')
    if osp.isfile(outfile):
        print(f'{outfile} already exists, skipping')
        return

    # load model
    from mmpose.apis import init_model, inference_topdown
    model = init_model(args.config, args.checkpoint, device=args.device)
    print('Sapiens model loaded.')

    # load video
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f'Cannot open video: {args.video}'
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # load masks for bounding boxes
    mask_file = osp.join(args.masks_root, f'{seq_name}_masks_k0.h5')
    h5 = h5py.File(mask_file, 'r')
    mask_group = h5[seq_name]

    all_keypoints = []
    frames_list = []

    for fid in tqdm(range(num_frames), desc='Sapiens pose'):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_id = f'{fid:06d}'
        frames_list.append(frame_id)

        # get person bbox from mask
        mask_key = f'{frame_id}-k0.person_mask.png'
        if mask_key not in mask_group:
            # no mask for this frame, fill with zeros
            all_keypoints.append(np.zeros((17, 3), dtype=np.float64))
            continue

        person_mask = mask_group[mask_key][:].astype(np.uint8)
        bbox = get_person_bbox_from_mask(person_mask)
        if bbox is None:
            all_keypoints.append(np.zeros((17, 3), dtype=np.float64))
            continue

        # Sapiens expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # run inference (top-down with bbox)
        results = inference_topdown(model, frame_rgb, bboxes=bbox[None], bbox_format='xyxy')

        if len(results) > 0 and hasattr(results[0], 'pred_instances'):
            kpts = results[0].pred_instances.keypoints[0]  # (17, 2)
            scores = results[0].pred_instances.keypoint_scores[0]  # (17,)
            kpt_with_conf = np.concatenate([kpts, scores[:, None]], axis=1)  # (17, 3)
            all_keypoints.append(kpt_with_conf.astype(np.float64))
        else:
            all_keypoints.append(np.zeros((17, 3), dtype=np.float64))

    cap.release()
    h5.close()

    # stack and add view dimension: (N, 17, 3) -> (N, 1, 17, 3)
    joints2d = np.stack(all_keypoints, axis=0)[:, None, :, :]  # (N, 1, 17, 3)
    print(f'Keypoints shape: {joints2d.shape}, frames: {len(frames_list)}')

    pack_data = {
        'frames': frames_list,
        'joints2d': joints2d,
    }
    joblib.dump(pack_data, outfile)
    print(f'Saved to {outfile}')


if __name__ == '__main__':
    main()
