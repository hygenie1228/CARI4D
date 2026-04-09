"""
COCO 17-keypoint body landmarks from SMPL mesh vertices.
Mirrors body_landmark.py but uses J_regressor_coco.npy (17 joints x 6890 verts).

COCO keypoint order:
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""
import numpy as np
from os.path import join
import torch


class CocoBodyLandmarks:
    "SMPL wrapper to compute COCO 17 body landmarks from SMPL mesh vertices"

    def __init__(self, assets_root):
        reg = np.load(join(assets_root, 'J_regressor_coco.npy'))  # (17, 6890)
        self.coco_reg = reg
        self.reg_body_th = torch.from_numpy(reg).float()

    def get_body_kpts(self, smpl_mesh):
        "return 17 COCO body keypoints from a single mesh"
        return self.coco_reg @ smpl_mesh.v  # (17, 3)

    def get_body_kpts_batch(self, smpl_verts):
        """
        Args:
            smpl_verts: np array, (B, N, 3)
        Returns: (B, 17, 3), numpy array
        """
        reg_th = self.reg_body_th.unsqueeze(0).expand(smpl_verts.shape[0], -1, -1)
        verts_th = torch.from_numpy(smpl_verts).float()
        J = torch.bmm(reg_th, verts_th)
        return J.numpy()

    def get_body_kpts_batch_torch(self, smpl_verts):
        """
        Args:
            smpl_verts: tensor, (B, N, 3)
        Returns: tensor, (B, 17, 3)
        """
        reg = self.reg_body_th.to(smpl_verts.device)
        reg = reg.unsqueeze(0).expand(smpl_verts.shape[0], -1, -1)
        return torch.bmm(reg, smpl_verts)
