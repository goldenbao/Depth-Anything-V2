"""Fuse ToF sparse point cloud with scaled mono depth → dense point cloud."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from depth_anything_v2.util.depth_eval import (
    valid_mask, ransac_reciprocal_fit, visualize,
    back_project_to_pointcloud, write_ply_ascii,
    rgb_guided_residual_interpolation,
)

# --- Paths ---
MATCH_DIR = Path("data/benchmark/match_data")
TOF_DIR = MATCH_DIR / "tof"
RGB_DIR = MATCH_DIR / "rgb_left"
MONO_DIR = Path("depth_vitl/benchmark")
OUT_DIR = Path("tof_fusion_vitl")
PLY_DIR = OUT_DIR / "ply"
VIS_DIR = OUT_DIR / "vis"

MAX_DEPTH = 0.6
MIN_DEPTH = 0.05
FX = FY = 293.756744
CX = 324.838501
CY = 241.260452
W, H = 640, 480

# cam2tof extrinsics
_R = np.array([
    [0.004059, -0.001540,  0.999991],
    [-0.999847, -0.017029,  0.004032],
    [0.017023,  -0.999854, -0.001609],
])
_t = np.array([0.165400, 0.032500, 0.062150])

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLY_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


# === ToF utilities (same as tof_vs_mono.py) ===

def load_tof_ply(path):
    pts = []
    with open(path) as f:
        in_header = True
        for line in f:
            if in_header:
                if line.startswith("end_header"):
                    in_header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(pts, dtype=np.float32)


def tof_to_camera(pts_tof):
    return (_R.T @ (pts_tof - _t).T).T


def project_to_depth(xyz_cam):
    mask = xyz_cam[:, 2] > 1e-6
    x, y, z = xyz_cam[mask].T
    u = FX * x / z + CX
    v = FY * y / z + CY
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inside], v[inside], z[inside]
    depth = np.full((H, W), np.nan, dtype=np.float32)
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[inside], vi[inside], z[inside]
    depth[vi, ui] = z
    return depth, ui, vi


def load_rgb(path):
    import cv2
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# === Build paired list ===
tof_pairs = []
for tof_f, left_f in zip(sorted(os.listdir(TOF_DIR)), sorted(os.listdir(RGB_DIR))):
    mono_f = left_f.replace(".jpg", "_raw_depth.npy")
    if not (MONO_DIR / mono_f).exists():
        continue
    tof_pairs.append((tof_f, left_f, mono_f))

print(f"Matched pairs: {len(tof_pairs)}")


# === Main fusion loop ===
all_mae_before, all_mae_after = [], []
all_rmse_before, all_rmse_after = [], []


# for tof_f, left_f, mono_f in tof_pairs[:20]:
    
# targets = {"SLAM_SLAM_L_TX0_83940_640X480.jpg", "SLAM_SLAM_L_TX0_131780_640X480.jpg"}
# tof_pairs = [(t, l, m) for t, l, m in tof_pairs if l in targets]
# print(f"Targets: {targets}, found: {len(tof_pairs)} pairs")

for tof_f, left_f, mono_f in tof_pairs:
    # Load data
    pts_tof = load_tof_ply(TOF_DIR / tof_f)
    pts_cam = tof_to_camera(pts_tof)
    gt_depth, ui, vi = project_to_depth(pts_cam)
    mono = np.load(MONO_DIR / mono_f, allow_pickle=False).astype(np.float32)

    rgb_path = RGB_DIR / left_f
    if not rgb_path.exists():
        continue
    rgb = load_rgb(rgb_path)

    # Reciprocal fit (at ToF pixels only)
    mask_fit = valid_mask(mono, gt_depth, MAX_DEPTH, drop_border=True)
    mv = mono[mask_fit]
    gv = gt_depth[mask_fit]
    if len(mv) < 10:
        continue

    a, b, n_in = ransac_reciprocal_fit(mv, gv)
    aligned = 1.0 / (a * mono + b)

    # Evaluate before fusion (at ToF pixels)
    mask_aligned = valid_mask(aligned, gt_depth, MAX_DEPTH, filter_mono=True, drop_border=True)
    err_before = np.abs(aligned[mask_aligned] - gt_depth[mask_aligned])
    mae_before = err_before.mean()
    rmse_before = np.sqrt((err_before ** 2).mean())

    # Compute sparse residual at projected ToF pixel locations
    # ui, vi are from project_to_depth (already on integer grid)
    valid_tof = (gt_depth[vi, ui] > MIN_DEPTH) & (gt_depth[vi, ui] < MAX_DEPTH) & ~np.isnan(gt_depth[vi, ui])
    if valid_tof.sum() < 10:
        continue

    u_sparse = ui[valid_tof]
    v_sparse = vi[valid_tof]
    residual_sparse = gt_depth[v_sparse, u_sparse] - aligned[v_sparse, u_sparse]

    # RGB-guided residual interpolation
    correction = rgb_guided_residual_interpolation(
        u_sparse, v_sparse, residual_sparse, rgb,
        sigma_spatial=20, sigma_color=30, k=16,
    )

    # Fuse
    fused = aligned + correction

    # Evaluate after fusion (at ToF pixels)
    mask_fused = valid_mask(fused, gt_depth, MAX_DEPTH, filter_mono=True, drop_border=True)
    err_after = np.abs(fused[mask_fused] - gt_depth[mask_fused])
    mae_after = err_after.mean()
    rmse_after = np.sqrt((err_after ** 2).mean())

    all_mae_before.append(mae_before)
    all_mae_after.append(mae_after)
    all_rmse_before.append(rmse_before)
    all_rmse_after.append(rmse_after)

    # Back-project aligned (pre-fusion) and fused point clouds
    base = left_f.replace(".jpg", "")

    # Mask out black border before back-projection
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[19:466, :629] = True
    aligned = np.where(border_mask, aligned, 0)
    fused = np.where(border_mask, fused, 0)

    aligned_pts, aligned_colors = back_project_to_pointcloud(aligned, rgb, FX, FY, CX, CY, max_depth=MAX_DEPTH)
    aligned_pts = (_R @ aligned_pts.T).T + _t
    write_ply_ascii(PLY_DIR / f"{base}_aligned.ply", aligned_pts, aligned_colors)

    fused_pts, colors = back_project_to_pointcloud(fused, rgb, FX, FY, CX, CY, max_depth=MAX_DEPTH)
    fused_pts = (_R @ fused_pts.T).T + _t
    write_ply_ascii(PLY_DIR / f"{base}_fused.ply", fused_pts, colors)

    # Diagnostic visualization
    diff_before = np.abs(aligned - gt_depth)
    diff_after = np.abs(fused - gt_depth)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax in axes.ravel():
        ax.clear()

    # (1) Aligned mono
    im = axes[0, 0].imshow(aligned, cmap="plasma", vmin=0, vmax=MAX_DEPTH)
    axes[0, 0].set_title("Aligned Mono")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    # (2) Fused
    im = axes[0, 1].imshow(fused, cmap="plasma", vmin=0, vmax=MAX_DEPTH)
    axes[0, 1].set_title("Fused (mono + ToF correction)")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # (3) Sparse GT
    vis_gt = np.full((H, W), np.nan)
    vis_gt[mask_aligned] = gt_depth[mask_aligned]
    im = axes[0, 2].imshow(vis_gt, cmap="plasma", vmin=0, vmax=MAX_DEPTH)
    axes[0, 2].set_title("ToF GT (sparse)")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # (4) Error before
    im = axes[1, 0].imshow(diff_before, cmap="hot", vmin=0, vmax=0.1)
    axes[1, 0].set_title(f"Error Before (MAE={mae_before:.4f})")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # (5) Error after
    im = axes[1, 1].imshow(diff_after, cmap="hot", vmin=0, vmax=0.1)
    axes[1, 1].set_title(f"Error After (MAE={mae_after:.4f})")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # (6) Scatter: before (blue) vs after (red) vs GT
    n_show = min(5000, mask_aligned.sum())
    idx_show = np.random.choice(mask_aligned.sum(), n_show, replace=False)
    gt_vals = gt_depth[mask_aligned][idx_show]
    aligned_vals = aligned[mask_aligned][idx_show]
    fused_vals = fused[mask_aligned][idx_show]

    axes[1, 2].scatter(gt_vals, aligned_vals, s=1, alpha=0.3, label="before", c="blue")
    axes[1, 2].scatter(gt_vals, fused_vals, s=1, alpha=0.3, label="after", c="red")
    lo = min(gt_vals.min(), aligned_vals.min(), fused_vals.min())
    hi = max(gt_vals.max(), aligned_vals.max(), fused_vals.max())
    axes[1, 2].plot([lo, hi], [lo, hi], "k--", lw=0.5)
    axes[1, 2].set_xlabel("ToF GT (m)")
    axes[1, 2].set_ylabel("Predicted (m)")
    axes[1, 2].legend(fontsize=7)

    plt.tight_layout()
    vis_path = VIS_DIR / left_f.replace(".jpg", ".png")
    plt.savefig(vis_path, dpi=150)
    plt.close()

    print(f"  {left_f}:  MAE {mae_before:.4f}→{mae_after:.4f}  "
          f"RMSE {rmse_before:.4f}→{rmse_after:.4f}  "
          f"({valid_tof.sum()} pts → {len(fused_pts)} cloud pts)")


# === Summary ===
if all_mae_before:
    before_mae = np.mean(all_mae_before)
    after_mae = np.mean(all_mae_after)
    before_rmse = np.mean(all_rmse_before)
    after_rmse = np.mean(all_rmse_after)
    improvement = (before_mae - after_mae) / before_mae * 100

    print(f"\n{'='*60}")
    print(f"Fusion Summary ({len(all_mae_before)} images)")
    print(f"{'='*60}")
    print(f"  MAE:  {before_mae:.4f} → {after_mae:.4f}  ({improvement:+.1f}%)")
    print(f"  RMSE: {before_rmse:.4f} → {after_rmse:.4f}")

    # Save summary
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(f"Fusion Summary ({len(all_mae_before)} images)\n")
        f.write(f"{'='*50}\n")
        f.write(f"MAE before: {before_mae:.4f}\n")
        f.write(f"MAE after:  {after_mae:.4f}\n")
        f.write(f"Improvement: {improvement:+.1f}%\n")
        f.write(f"RMSE before: {before_rmse:.4f}\n")
        f.write(f"RMSE after:  {after_rmse:.4f}\n")
