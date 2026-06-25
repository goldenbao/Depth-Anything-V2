import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from depth_anything_v2.util.depth_eval import load_npy, valid_mask, visualize, ransac_reciprocal_fit

stereo_dir = "data/benchmark/match_data/depth"

mono_dir   = "depth_vitl/benchmark"
out_dir    = "./depth_compare_vitl/benchmark"


MAX_DEPTH = 0.6
FIT_METHOD = 'lstsq'   # 'lstsq' or 'ransac'

os.makedirs(out_dir, exist_ok=True)

files = sorted(os.listdir(stereo_dir))

#region  # === Pass 1: collect all valid pixel values ===
# all_mono_pool = []
# all_stereo_pool = []

# for f in files:
#     stereo_path = os.path.join(stereo_dir, f)
#     mono_path = os.path.join(mono_dir, f.replace('.jpg','_raw_depth'))

#     if not os.path.exists(mono_path):
#         continue

#     stereo = load_npy(stereo_path)
#     mono = load_npy(mono_path)

#     if stereo.shape != mono.shape:
#         print(f"skip (shape mismatch): {f}")
#         continue

#     mask = valid_mask(mono, stereo, MAX_DEPTH, drop_border=True)
#     all_mono_pool.append(mono[mask])
#     all_stereo_pool.append(stereo[mask])

# if not all_mono_pool:
#     print("No valid data found.")
#     exit()

# all_mono_pool = np.concatenate(all_mono_pool)
# all_stereo_pool = np.concatenate(all_stereo_pool)

# # Global least squares fit: inv_stereo = a * mono + b
# inv_stereo = 1.0 / all_stereo_pool
# A = np.stack([all_mono_pool, np.ones_like(all_mono_pool)], axis=1)
# global_a, global_b = np.linalg.lstsq(A, inv_stereo, rcond=None)[0]
# print(f"\nGlobal fit:  1/gt = {global_a:.6f} * mono + {global_b:.6f}")

# # Scatter: mono vs 1/stereo (subsample)
# idx = np.random.choice(len(all_mono_pool), size=min(50000, len(all_mono_pool)), replace=False)
# inv_stereo_sample = 1.0 / all_stereo_pool[idx]
# plt.figure(figsize=(6, 6))
# plt.scatter(all_mono_pool[idx], inv_stereo_sample, s=1, alpha=0.3)
# plt.xlabel("Mono (raw)")
# plt.ylabel("1 / Stereo (ground truth)")
# plt.title(f"Mono vs Inverse Stereo (points < {MAX_DEPTH:.1f}m)")
# x_line = np.linspace(np.percentile(all_mono_pool, 1), np.percentile(all_mono_pool, 99), 100)
# plt.plot(x_line, global_a * x_line + global_b, 'r--',
#          label=f"fit: 1/gt={global_a:.4f}·mono+{global_b:.4f}")
# # focus on dense region
# x_lo, x_hi = np.percentile(all_mono_pool[idx], [1, 99])
# y_lo, y_hi = np.percentile(inv_stereo_sample, [1, 99])
# plt.xlim(x_lo, x_hi)
# plt.ylim(y_lo, y_hi)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "scatter_mono_vs_inv_stereo.png"))
# # plt.show()
# plt.close()

# # Scatter: predicted depth (from global reciprocal fit) vs stereo ground truth
# pred_depth = 1.0 / (global_a * all_mono_pool + global_b)
# plt.figure(figsize=(6, 6))
# plt.scatter(pred_depth[idx], all_stereo_pool[idx], s=1, alpha=0.3)
# plt.xlabel("Predicted depth = 1/(a·mono + b)")
# plt.ylabel("Stereo ground truth")
# plt.title(f"Predicted vs GT Depth (points < {MAX_DEPTH:.1f}m)")
# x_lo, x_hi = np.percentile(pred_depth[idx], [1, 99])
# y_lo, y_hi = np.percentile(all_stereo_pool[idx], [1, 99])
# lo = min(x_lo, y_lo)
# hi = max(x_hi, y_hi)
# plt.plot([lo, hi], [lo, hi], 'r--', label="y = x")
# plt.xlim(lo, hi)
# plt.ylim(lo, hi)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "scatter_pred_vs_gt_depth.png"))
# # plt.show()
# plt.close()

# # Global prediction error stats
# pred_global = 1.0 / (global_a * all_mono_pool + global_b)
# diff_global = np.abs(pred_global - all_stereo_pool)
# print(f"\nGlobal fit error (on {len(all_mono_pool)} points):")
# print(f"  MAE={diff_global.mean():.4f}  RMSE={np.sqrt((diff_global**2).mean()):.4f}")
# print(f"  min={diff_global.min():.4f}   max={diff_global.max():.4f}")
#endregion


# === Pass 2: per-image reciprocal fit ===
all_mae = []
all_rmse = []
all_min = []
all_max = []
all_max_file = []

for f in files:
    # if "132580" not in f:
    #     continue
    stereo_path = os.path.join(stereo_dir, f)
    mono_path = os.path.join(mono_dir, f.replace('.jpg','_raw_depth'))

    if not os.path.exists(mono_path):
        continue

    stereo = load_npy(stereo_path)
    mono = load_npy(mono_path)

    if stereo.shape != mono.shape:
        continue

    mask = valid_mask(mono, stereo, MAX_DEPTH, drop_border=True)
    mono_v = mono[mask]
    stereo_v = stereo[mask]

    # Fit1: 1/gt = a * mono + b
    a1, b1, _ = ransac_reciprocal_fit(mono_v, stereo_v, method=FIT_METHOD)
    print(f"  {f}:  Fit1 1/gt=a·mono+b:  {a1:.6f}·mono+{b1:.6f}")
    aligned = 1.0 / (a1 * mono + b1)
    
    aligned_mask = valid_mask(aligned, stereo, MAX_DEPTH, filter_mono=True, drop_border=True)

    #region # Fit2: gt = a * (1/mono) + b
    # inv_mono_v = 1.0 / mono_v
    # # FIT_METHOD == 'ransac'
    # if FIT_METHOD == 'ransac':
    #     ransac2 = RANSACRegressor(residual_threshold=0.05, max_trials=100)
    #     ransac2.fit(inv_mono_v.reshape(-1, 1), stereo_v)
    #     a2 = ransac2.estimator_.coef_[0]
    #     b2 = ransac2.estimator_.intercept_
    # else:
    #     A2 = np.stack([inv_mono_v, np.ones_like(inv_mono_v)], axis=1)
    #     a2, b2 = np.linalg.lstsq(A2, stereo_v, rcond=None)[0]
    # print(f"  {f}:  Fit2 gt=a·(1/mono)+b: gt={a2:.6f}·(1/mono)+{b2:.6f}")

    # # Align with both fits
    # aligned1 = 1.0 / (a1 * mono + b1)
    # aligned2 = a2 / mono + b2

    # aligned_mask = valid_mask(aligned1, stereo, MAX_DEPTH, filter_mono=True, drop_border=True)
    # err1 = np.abs(aligned1[aligned_mask] - stereo[aligned_mask])
    # err2 = np.abs(aligned2[aligned_mask] - stereo[aligned_mask])
    # mae1, rmse1 = err1.mean(), np.sqrt((err1 ** 2).mean())
    # mae2, rmse2 = err2.mean(), np.sqrt((err2 ** 2).mean())
    # best = "Fit1" if mae1 <= mae2 else "Fit2"
    # print(f"  {f}:  Fit1 MAE={mae1:.4f} RMSE={rmse1:.4f} max={err1.max():.3f}  |  \
    #     Fit2 MAE={mae2:.4f} RMSE={rmse2:.4f} max={err2.max():.3f}  →  {best}")
    # Use the better fit for downstream
    # if mae1 <= mae2:
    #     mono_aligned = aligned1
    #     a_use, b_use = a1, b1
    # else:
    #     mono_aligned = aligned2
    #     a_use, b_use = a2, b2
    #endregion
    
    aligned_gt = stereo[aligned_mask]
    aligned_mask= aligned[aligned_mask]
    depth_errors = np.abs(aligned_mask - aligned_gt)

    all_mae.append(depth_errors.mean())
    all_rmse.append(np.sqrt((depth_errors ** 2).mean()))
    all_min.append(depth_errors.min())
    all_max.append(depth_errors.max())
    all_max_file.append(f)

    diff = np.abs(aligned - stereo)

    vis_path = os.path.join(out_dir, f.replace(".npy", ".png"))
    visualize(aligned, stereo, diff, vis_path, gt_label="Stereo Depth", drop_border=True,
              mono_raw=mono, fit_a=a1, fit_b=b1)

    #region two methods scatter plot
    # Scatter: Fit1 (mono vs 1/gt) with fitted line
    # n_scatter = min(20000, len(mono_v))
    # idx_scatter = np.random.choice(len(mono_v), n_scatter, replace=False)
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # inv_gt = 1.0 / stereo_v
    # axes[0].scatter(mono_v[idx_scatter], inv_gt[idx_scatter], s=1, alpha=0.3)
    # x_line = np.linspace(mono_v.min(), mono_v.max(), 100)
    # axes[0].plot(x_line, a1 * x_line + b1, "r-", lw=1,
    #              label=f"1/gt={a1:.4f}·mono+{b1:.4f}")
    # axes[0].set_xlabel("mono depth")
    # axes[0].set_ylabel("1 / GT")
    # axes[0].set_title(f"Fit1: 1/gt=a·mono+b")
    # axes[0].legend(fontsize=8)
    # axes[0].grid(True, alpha=0.3)

    # # Scatter: Fit2 (1/mono vs gt) with fitted line
    # inv_mono = 1.0 / mono_v
    # axes[1].scatter(inv_mono[idx_scatter], stereo_v[idx_scatter], s=1, alpha=0.3)
    # x_lo, x_hi = np.percentile(inv_mono[idx_scatter], [1, 99])
    # x_line2 = np.linspace(x_lo, x_hi, 100)
    # axes[1].set_xlim(x_lo, x_hi)
    # axes[1].plot(x_line2, a2 * x_line2 + b2, "r-", lw=1,
    #              label=f"gt={a2:.4f}·(1/mono)+{b2:.4f}")
    # axes[1].set_xlabel("1 / mono depth")
    # axes[1].set_ylabel("GT depth")
    # axes[1].set_title(f"Fit2: gt=a·(1/mono)+b")
    # axes[1].legend(fontsize=8)
    # axes[1].grid(True, alpha=0.3)

    # plt.tight_layout()
    # scatter_path = os.path.join(out_dir, f.replace(".npy", "_fit_compare.png"))
    # plt.savefig(scatter_path, dpi=150)
    # plt.show()
    # plt.close()
    #endregion

    # standalone interactive view
    plt.figure(figsize=(12, 10))
    plt.imshow(aligned, cmap="plasma", vmin=0, vmax=1)
    plt.colorbar(label="Depth (m)")
    plt.title(f"{f} - Aligned Mono Depth")
    plt.tight_layout()
    plt.show()
    plt.close()


print(f"\n==== Summary (reciprocal fit, {len(all_mae)} images) ====")
print(f"  MAE:  mean={np.mean(all_mae):.4f}")
print(f"  RMSE: mean={np.mean(all_rmse):.4f}")
print(f"  Min depth error (global): {min(all_min):.4f}")
max_err = max(all_max)
max_idx = all_max.index(max_err)
print(f"  Max depth error (global): {max_err:.4f}  (file: {all_max_file[max_idx]})")
