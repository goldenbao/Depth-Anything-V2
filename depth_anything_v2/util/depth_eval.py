"""Shared utilities for mono depth evaluation against ground truth."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MIN_DEPTH =1e-6


def load_npy(path):
    return np.load(path, allow_pickle=False).astype(np.float32)


def valid_mask(pred, gt, max_depth=None, filter_mono=False, drop_border=False):
    """Boolean mask where both pred and gt have valid values.

    When drop_border=True, excludes black-border regions:
    rows 0-18, rows 466-479, and cols 629-639 (for 480×640 images).
    """
    mask = (pred > MIN_DEPTH) & (gt > MIN_DEPTH) & np.isfinite(pred) & np.isfinite(gt)
    if max_depth is not None:
        mask = mask & (gt < max_depth)
        if filter_mono:
            mask = mask & (pred < max_depth)
    if drop_border:
        H, W = pred.shape[:2]
        valid_region = np.zeros((H, W), dtype=bool)
        valid_region[19:466, :629] = True
        mask = mask & valid_region
    return mask


def ransac_reciprocal_fit(mono, gt, threshold=0.3, max_trials=100, method='lstsq'):
    """Fit 1/gt = a * mono + b.

    method='lstsq':  ordinary least squares, returns (a, b, None).
    method='ransac': RANSAC robust fit, returns (a, b, n_inliers).
    """
    inv_gt = 1.0 / gt
    if method == 'lstsq':
        A = np.stack([mono, np.ones_like(mono)], axis=1)
        a, b = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
        return a, b, None
    from sklearn.linear_model import RANSACRegressor
    ransac = RANSACRegressor(residual_threshold=threshold, max_trials=max_trials)
    ransac.fit(mono.reshape(-1, 1), inv_gt)
    a = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_
    return a, b, ransac.inlier_mask_.sum()


def back_project_to_pointcloud(depth, rgb, fx, fy, cx, cy, max_depth=2.0):
    """Back-project depth map to 3D point cloud. Returns (N×3 points, N×3 colors uint8)."""
    H, W = depth.shape[:2]
    mask = (depth > MIN_DEPTH) & (depth < max_depth) & np.isfinite(depth)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u[mask] - cx) * depth[mask] / fx
    y = (v[mask] - cy) * depth[mask] / fy
    z = depth[mask]
    points = np.stack([x, y, z], axis=1)
    colors = rgb[mask]
    return points, colors


def write_ply_ascii(path, points, colors):
    """Write N×3 points + N×3 uint8 colors to ASCII PLY file."""
    N = len(points)
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(N):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")


def rgb_guided_residual_interpolation(sparse_u, sparse_v, sparse_residual, rgb,
                                       sigma_spatial=20, sigma_color=30, k=16):
    """Interpolate sparse residuals to dense (H,W) using RGB-guided bilateral weights.

    Uses scipy KDTree for fast nearest-neighbor lookup.
    """
    from scipy.spatial import KDTree
    H, W = rgb.shape[:2]
    rgb_f = rgb.astype(np.float32)

    # Build KDTree from sparse pixel coordinates
    coords = np.stack([sparse_v, sparse_u], axis=1)  # (row, col)
    tree = KDTree(coords)

    # Query for every pixel
    grid_v, grid_u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    grid_coords = np.stack([grid_v.ravel(), grid_u.ravel()], axis=1)

    dists, idxs = tree.query(grid_coords, k=min(k, len(sparse_u)))

    # Spatial weights
    w_spatial = np.exp(-0.5 * (dists / sigma_spatial) ** 2)

    # Color weights: rgb at each pixel vs rgb at each of its k neighbors
    neighbor_colors = rgb_f[coords[idxs, 0], coords[idxs, 1]]  # (N_pixels, k, 3)
    pixel_colors = rgb_f[grid_v.ravel(), grid_u.ravel()]        # (N_pixels, 3)
    color_diff = neighbor_colors - pixel_colors[:, None, :]
    w_color = np.exp(-0.5 * ((color_diff / sigma_color) ** 2).sum(axis=2))

    w = w_spatial * w_color
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.clip(w_sum, 1e-10, None)

    neighbor_residuals = sparse_residual[idxs]  # (N_pixels, k)
    dense_residual = (w * neighbor_residuals).sum(axis=1).reshape(H, W)

    # Clip to 3x RMS of sparse residuals
    rms = np.sqrt(np.mean(sparse_residual ** 2))
    dense_residual = np.clip(dense_residual, -3 * rms, 3 * rms)

    return dense_residual.astype(np.float32)


def visualize(aligned, gt, diff, save_path, *, sparse=False, max_depth=0.6,
              gt_label="GT", drop_border=False, mono_raw=None, fit_a=None, fit_b=None):
    """3x3 evaluation figure: mono, GT, diff, scatter, histogram, error overlay.

    When sparse=True, GT/diff are NaN-masked (only valid pixels shown).
    When mono_raw is provided, adds a 7th subplot: 1/gt vs mono_raw scatter.
    """
    H, W = gt.shape[:2]
    mask = valid_mask(aligned, gt, max_depth, filter_mono=True, drop_border=drop_border)
    mv = aligned[mask]
    gv = gt[mask]
    dv = diff[mask]

    if len(mv) == 0:
        return

    n = min(20000, len(mv))
    idx = np.random.choice(len(mv), n, replace=False)

    plt.figure(figsize=(18, 12))

    # (1) Mono aligned
    plt.subplot(3, 3, 1)
    plt.title("Mono (aligned)")
    plt.imshow(aligned, cmap="plasma", vmin=0, vmax=max_depth)
    plt.colorbar()

    # (2) GT
    plt.subplot(3, 3, 2)
    plt.title(gt_label)
    if sparse:
        vis = np.full((H, W), np.nan)
        vis[mask] = gv
        plt.imshow(vis, cmap="plasma", vmin=0, vmax=max_depth)
    else:
        plt.imshow(gt, cmap="plasma", vmin=0, vmax=max_depth)
    plt.colorbar()

    # (3) Abs Diff
    diff_vmax = 0.1 if sparse else max_depth
    plt.subplot(3, 3, 3)
    plt.title("Abs Diff")
    if sparse:
        vis = np.full((H, W), np.nan)
        vis[mask] = dv
        plt.imshow(vis, cmap="hot", vmin=0, vmax=diff_vmax)
    else:
        plt.imshow(diff, cmap="hot", vmin=0, vmax=diff_vmax)
    plt.colorbar()

    # (4) Scatter (aligned)
    plt.subplot(3, 3, 4)
    plt.scatter(mv[idx], gv[idx], s=1, alpha=0.3)
    plt.xlabel("Aligned pred depth")
    plt.ylabel(gt_label)
    lo = min(mv.min(), gv.min())
    hi = max(mv.max(), gv.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="y=x")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.legend(fontsize=8)

    # (5) Error histogram
    plt.subplot(3, 3, 5)
    bins = np.linspace(0, min(max_depth, dv.max()), 100)
    plt.hist(dv, bins=bins, color="steelblue", edgecolor="none", alpha=0.7)
    plt.xlabel("Abs error (m)")
    plt.ylabel("Pixel count")
    plt.yscale("log")
    mean_err = dv.mean()
    plt.axvline(mean_err, color="r", linestyle="--", label=f"mean={mean_err:.4f}")
    for p in [50, 90, 95, 99]:
        val = np.percentile(dv, p)
        plt.axvline(val, color="gray", linestyle=":", alpha=0.5)
        plt.text(val, plt.ylim()[1] * 0.9, f"{p}%", fontsize=7, rotation=90,
                 va="top", ha="right")
    plt.legend(fontsize=8)

    # (6) Error overlay
    ol_edges = [0, 0.005, 0.02, 0.05, 0.1]
    if sparse:
        ol_edges.append(max_depth)
    ol_colors = np.array([
        [0, 0, 0],       # no data / masked
        [0, 0.8, 0],     # < 0.005
        [0.8, 0.8, 0],   # 0.005-0.02
        [1, 0.5, 0],     # 0.02-0.05
        [1, 0, 0],       # 0.05-0.1
        [0.5, 0, 0],     # > 0.1 (dense) / 0.1-0.6 (sparse)
    ])
    if sparse:
        ol_colors = np.vstack([ol_colors, [[0.2, 0, 0.2]]])  # >= max_depth
    ol_labels = ["masked", "<0.005", "0.005-0.02", "0.02-0.05", "0.05-0.1"]
    ol_labels += [">=0.1" if not sparse else "0.1-MAX"]
    if sparse:
        ol_labels.append(">=MAX")

    overlay = np.zeros((H, W, 3))
    cats = np.zeros((H, W), dtype=int)
    cats[mask] = np.digitize(dv, ol_edges)
    for c in range(len(ol_edges) + 1):
        overlay[cats == c] = ol_colors[c]

    plt.subplot(3, 3, 6)
    plt.imshow(overlay)
    plt.title("Error Overlay")
    legend_elements = [Patch(color=c, label=l) for c, l in zip(ol_colors, ol_labels)]
    plt.legend(handles=legend_elements, fontsize=6, loc="upper right")

    # (7) 1/gt vs mono_raw (pre-alignment)
    if mono_raw is not None:
        raw_masked = mono_raw[mask]
        inv_gt = 1.0 / gv
        plt.subplot(3, 3, 7)
        plt.scatter(raw_masked[idx], inv_gt[idx], s=1, alpha=0.3)
        plt.xlabel("Mono raw depth")
        plt.ylabel("1 / GT")
        if fit_a is not None and fit_b is not None:
            x_line = np.linspace(raw_masked.min(), raw_masked.max(), 100)
            plt.plot(x_line, fit_a * x_line + fit_b, "r-", lw=1,
                     label=f"y={fit_a:.4f}x+{fit_b:.4f}")
            plt.legend(fontsize=8)
        plt.title("1/GT vs mono raw (pre-alignment)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
