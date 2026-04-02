import numpy as np
import open3d as o3d
from numpy.lib.stride_tricks import sliding_window_view


def bilateral_filter_depth_cpu_fast(
    depth: np.ndarray,
    radius: int = 2,
    zfar: float = 100.0,
    sigmaD: float = 2.0,
    sigmaR: float = 100000.0,
) -> np.ndarray:
    """
    Vectorized CPU depth bilateral filter.
    """
    assert depth.ndim == 2
    depth = np.asarray(depth, dtype=np.float32, order="C")
    h, w = depth.shape
    k = 2 * radius + 1

    depth_pad = np.pad(depth, radius, mode="constant", constant_values=0.0)
    valid = (depth >= 0.001) & (depth < zfar)
    valid_pad = np.pad(valid.astype(np.uint8), radius, mode="constant", constant_values=0)

    dp = sliding_window_view(depth_pad, (k, k))
    vp = sliding_window_view(valid_pad, (k, k)).astype(bool)

    num_valid = vp.sum(axis=(2, 3)).astype(np.float32)
    sum_valid = (dp * vp).sum(axis=(2, 3), dtype=np.float32)
    mean_valid = sum_valid / np.maximum(num_valid, 1.0)

    ys = np.arange(-radius, radius + 1, dtype=np.float32)
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    spatial = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * sigmaD * sigmaD)).astype(np.float32)

    gate = vp & (np.abs(dp - mean_valid[..., None, None]) < 0.01)
    center = depth[..., None, None]
    range_w = np.exp(-((center - dp) ** 2) / (2.0 * sigmaR * sigmaR)).astype(np.float32)

    weights = range_w * spatial[None, None, ...]
    weights = np.where(gate, weights, 0.0).astype(np.float32)
    sum_w = weights.sum(axis=(2, 3), dtype=np.float32)

    num = (weights * dp).sum(axis=(2, 3), dtype=np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    valid_out = (sum_w > 0.0) & (num_valid > 0.0)
    out[valid_out] = (num[valid_out] / sum_w[valid_out]).astype(np.float32)
    return out


def erode_depth_cpu_fast(
    depth: np.ndarray,
    radius: int = 2,
    depth_diff_thres: float = 0.001,
    ratio_thres: float = 0.8,
    zfar: float = 100.0,
) -> np.ndarray:
    """
    Vectorized CPU depth erosion for outlier removal.
    """
    assert depth.ndim == 2
    depth = np.asarray(depth, dtype=np.float32, order="C")
    h, w = depth.shape
    k = 2 * radius + 1

    depth_pad = np.pad(depth, radius, mode="constant", constant_values=0.0)
    dp = sliding_window_view(depth_pad, (k, k))

    ones = np.ones((h, w), dtype=np.uint8)
    ones_pad = np.pad(ones, radius, mode="constant", constant_values=0)
    totals = sliding_window_view(ones_pad, (k, k)).sum(axis=(2, 3)).astype(np.float32)

    center = depth[..., None, None]
    bad = (dp < 0.001) | (dp >= zfar) | (np.abs(dp - center) > depth_diff_thres)
    bad_count = bad.sum(axis=(2, 3)).astype(np.float32)

    ratio = bad_count / np.maximum(totals, 1.0)
    center_invalid = (depth < 0.001) | (depth >= zfar)
    out = np.where(center_invalid | (ratio > ratio_thres), 0.0, depth).astype(np.float32)
    return out


def translation_only_icp_torch(
    src,
    tgt,
    R_fixed=np.eye(3),
    voxel_size=0.005,
    max_iter=30,
    tol=0.001,
    max_iters=[15, 15, 15],
):
    """
    Translation-only ICP (z-axis update) with Open3D downsampling + PyTorch3D KNN.
    """
    del max_iter  # kept for API compatibility
    import torch
    from pytorch3d.ops import knn_points

    src_work = o3d.geometry.PointCloud(src)
    tgt_work = o3d.geometry.PointCloud(tgt)
    src_work.rotate(R_fixed, center=(0, 0, 0))

    total_t = np.zeros(3)
    voxel_radius = [voxel_size * 8, voxel_size * 4, voxel_size]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for it, radius in zip(max_iters, voxel_radius):
        src_down = src_work.voxel_down_sample(voxel_size=voxel_size)
        tgt_down = tgt_work.voxel_down_sample(voxel_size=voxel_size)

        src_pts = torch.from_numpy(np.asarray(src_down.points)).float().to(device)
        tgt_pts = torch.from_numpy(np.asarray(tgt_down.points)).float().to(device)
        if src_pts.numel() == 0 or tgt_pts.numel() == 0:
            break

        thr2 = float((radius * 2) ** 2)
        total_t_it = np.zeros(3)

        for _ in range(it):
            dists, idx, _ = knn_points(src_pts.unsqueeze(0), tgt_pts.unsqueeze(0), K=1)
            d2 = dists[0, :, 0]
            nn_idx = idx[0, :, 0].long()

            mask = d2 < thr2
            if mask.sum().item() == 0:
                break

            tgt_matched = tgt_pts[nn_idx]
            res = tgt_matched[mask] - src_pts[mask]
            delta_t = res.mean(dim=0)
            delta_t[:2] = 0.0

            if torch.norm(delta_t).item() < tol:
                break

            src_pts = src_pts + delta_t
            delta_np = delta_t.detach().cpu().numpy()
            total_t = total_t + delta_np
            total_t_it += delta_np

        src_work.translate(total_t_it)

    t = np.eye(4)
    t[:3, :3] = R_fixed
    t[:3, 3] = total_t
    return t
