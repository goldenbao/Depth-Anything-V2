"""Calibrate mono depth vs ToF ground truth (sparse point cloud)."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from depth_anything_v2.util.depth_eval import valid_mask, visualize, ransac_reciprocal_fit

# --- Paths ---
MATCH_DIR = Path("data/benchmark/match_data")
TOF_DIR = MATCH_DIR / "tof"
RGB_DIR = MATCH_DIR / "rgb_left"
MONO_DIR = Path("depth_vitl/benchmark")
OUT_DIR = Path("tof_compare_vitl")

MAX_DEPTH = 0.6
FIT_METHOD = 'lstsq'   # 'lstsq' or 'ransac'
FX = FY = 293.756744
CX = 324.838501
CY = 241.260452
W, H = 640, 480

# cam2tof extrinsics: camera → ToF.  We need the inverse (ToF → camera).
_R = np.array([
    [0.004059, -0.001540,  0.999991],
    [-0.999847, -0.017029,  0.004032],
    [0.017023,  -0.999854, -0.001609],
])
_t = np.array([0.165400, 0.032500, 0.062150])

OUT_DIR.mkdir(parents=True, exist_ok=True)


# === Utility functions ===

def load_tof_ply(path):
    """Parse ASCII PLY file, return N×3 array of XYZ in ToF frame."""
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
    """Transform N×3 points from ToF frame to camera frame."""
    # cam2tof: P_tof = R @ P_cam + t  →  P_cam = R^T @ (P_tof - t)
    return (_R.T @ (pts_tof - _t).T).T


def project_to_depth(xyz_cam):
    """Project camera-frame points to image, return sparse depth map.
    Returns (sparse_depth, u_norm, v_norm) where u_norm/v_norm are 0..1 for plotting.
    """
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

    return depth, u / W, v / H   # normed for overlay plotting


# === Build ToF ↔ Mono mapping (already one-to-one aligned) ===
tof_pairs = []
for tof_f, left_f in zip(sorted(os.listdir(TOF_DIR)), sorted(os.listdir(RGB_DIR))):
    mono_f = left_f.replace(".jpg", "_raw_depth.npy")
    if not (MONO_DIR / mono_f).exists():
        continue
    tof_pairs.append((tof_f, left_f, mono_f))

print(f"Matched pairs: {len(tof_pairs)}")


# === Main loop ===
all_mae, all_rmse, all_min, all_max, all_files = [], [], [], [], []

for tof_f, left_f, mono_f in tof_pairs:
    # Load and transform ToF
    pts_tof = load_tof_ply(TOF_DIR / tof_f)
    pts_cam = tof_to_camera(pts_tof)
    gt_depth, _, _ = project_to_depth(pts_cam)

    # Load mono
    mono = np.load(MONO_DIR / mono_f, allow_pickle=False).astype(np.float32)

    # Sub-sample mono to match ToF points
    mask = valid_mask(mono, gt_depth, MAX_DEPTH, drop_border=True)
    mv = mono[mask]
    gv = gt_depth[mask]

    if len(mv) < 10:
        continue

    # Fit1 (least squares): 1/gt = a * mono + b
    a1, b1, _ = ransac_reciprocal_fit(mv, gv, method=FIT_METHOD)
    print(f"  Fit1 1/gt=a·mono+b:  1/gt = {a1:.6f}·mono + {b1:.6f}")
    mono_aligned1 = 1.0 / (a1 * mono + b1)
    aligned_mask = valid_mask(mono_aligned1, gt_depth, MAX_DEPTH, filter_mono=True, drop_border=True)
    
    #region Fit2 (least squares): gt = a2 * (1/mono) + b2
    # inv_mv = 1.0 / mv
    # if FIT_METHOD == 'ransac':
    #     from sklearn.linear_model import RANSACRegressor
    #     ransac2 = RANSACRegressor(residual_threshold=0.05, max_trials=100)
    #     ransac2.fit(inv_mv.reshape(-1, 1), gv)
    #     a2 = ransac2.estimator_.coef_[0]
    #     b2 = ransac2.estimator_.intercept_
    # else:
    #     A2 = np.stack([inv_mv, np.ones_like(inv_mv)], axis=1)
    #     a2, b2 = np.linalg.lstsq(A2, gv, rcond=None)[0]
    # print(f"  Fit2 gt=a·(1/mono)+b:  gt = {a2:.6f}·(1/mono) + {b2:.6f}")

    # mono_aligned2 = a2 / mono + b2

    # aligned_mask = valid_mask(mono_aligned1, gt_depth, MAX_DEPTH, filter_mono=True, drop_border=True)
    # err1 = np.abs(mono_aligned1[aligned_mask] - gt_depth[aligned_mask])
    # err2 = np.abs(mono_aligned2[aligned_mask] - gt_depth[aligned_mask])
    # mae1, rmse1 = err1.mean(), np.sqrt((err1 ** 2).mean())
    # mae2, rmse2 = err2.mean(), np.sqrt((err2 ** 2).mean())
    # best = "Fit1" if mae1 <= mae2 else "Fit2"
    # print(f"  Compare: Fit1 MAE={mae1:.4f} RMSE={rmse1:.4f}  |  Fit2 MAE={mae2:.4f} RMSE={rmse2:.4f}  →  {best}")

    # # Use the better fit for downstream
    # if mae1 <= mae2:
    #     mono_aligned = mono_aligned1
    #     a_use, b_use = a1, b1
    # else:
    #     mono_aligned = mono_aligned2
    #     a_use, b_use = a2, b2
    #endregion
        
        
    aligned_pred = mono_aligned1[aligned_mask]
    aligned_gt = gt_depth[aligned_mask]
    depth_errors = np.abs(aligned_pred - aligned_gt)

    all_mae.append(depth_errors.mean())
    all_rmse.append(np.sqrt((depth_errors ** 2).mean()))
    all_min.append(depth_errors.min())
    all_max.append(depth_errors.max())
    all_files.append(left_f)

    diff = np.abs(mono_aligned1 - gt_depth)
    vis_path = OUT_DIR / left_f.replace(".jpg", ".png")
    visualize(mono_aligned1, gt_depth, diff, vis_path, sparse=True, drop_border=True,
              mono_raw=mono, fit_a=a1, fit_b=b1)

    # Additional scatter: GT depth vs 1/mono depth
    # n_scatter = min(20000, len(mv))
    # idx_scatter = np.random.choice(len(mv), n_scatter, replace=False)
    # x_vals = 1.0 / mv[idx_scatter]
    # y_vals = gv[idx_scatter]
    # fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    # ax2.scatter(x_vals, y_vals, s=1, alpha=0.3)
    # x_lo, x_hi = np.percentile(x_vals, [1, 99])
    # ax2.set_xlim(x_lo, x_hi)
    # x_line = np.linspace(x_lo, x_hi, 200)
    # ax2.plot(x_line, a2 * x_line + b2, "r-", lw=1,
    #          label=f"gt={a2:.4f}·(1/mono)+{b2:.4f}")
    # ax2.set_xlabel("1 / mono depth")
    # ax2.set_ylabel("GT depth")
    # ax2.set_title("GT depth vs 1/mono depth")
    # ax2.legend(fontsize=8)
    # ax2.grid(True, alpha=0.3)
    # ax2.set_aspect("auto")
    # scatter_path = OUT_DIR / left_f.replace(".jpg", "_inv_scatter.png")
    # plt.tight_layout()
    # plt.savefig(scatter_path, dpi=150)
    # plt.show()
    # plt.close()

    # Interactive view
    # plt.figure(figsize=(12, 10))
    # mask_show = valid_mask(mono_aligned, gt_depth, MAX_DEPTH, drop_border=True)
    # vis = np.full((H, W), np.nan)
    # vis[mask_show] = mono_aligned[mask_show]
    # plt.imshow(vis, cmap="plasma", vmin=0, vmax=MAX_DEPTH)
    # plt.colorbar(label="Aligned depth (m)")
    # plt.title(f"{left_f} - Aligned Mono (ToF GT)")
    # plt.tight_layout()
    # plt.show()
    # plt.close()


# === Summary ===
if all_mae:
    print(f"\n==== Summary (ToF reciprocal fit, {len(all_mae)} images) ====")
    print(f"  MAE:  mean={np.mean(all_mae):.4f}")
    print(f"  RMSE: mean={np.mean(all_rmse):.4f}")
    print(f"  Min depth error (global): {min(all_min):.4f}")
    max_err = max(all_max)
    max_idx = all_max.index(max_err)
    print(f"  Max depth error (global): {max_err:.4f}  (file: {all_files[max_idx]})")
