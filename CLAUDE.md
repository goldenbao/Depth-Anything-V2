# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run relative depth estimation on images
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis

# Run relative depth estimation on videos
python run_video.py --encoder vitl --video-path assets/examples_video --outdir video_depth_vis

# Run Gradio web demo
python app.py

# Metric depth estimation (after downloading metric checkpoint)
cd metric_depth
python run.py --encoder vitl --load-from <checkpoint> --img-path <path> --outdir depth_vis

# Train metric depth model (distributed)
cd metric_depth
python -m torch.distributed.launch --nproc_per_node=<n> train.py --encoder vitl --dataset hypersim --epochs 40 --bs 2 --lr 5e-6 --save-path <path>

# Compare mono depth vs stereo ground truth
python scripts/stereo_vs.py

# Calibrate mono depth vs ToF (sparse point cloud)
python scripts/tof_vs_mono.py

# Fuse mono depth + ToF → dense point cloud (experimental)
python scripts/tof_mono_fusion.py
```

Encoders: `vits` | `vitb` | `vitl` | `vitg`
Common args: `--input-size <size>` (default 518), `--pred-only`, `--grayscale`

## Checkpoints

Download from Hugging Face and place in `checkpoints/`:
- `depth_anything_v2_vits.pth` (24.8M params)
- `depth_anything_v2_vitb.pth` (97.5M params)
- `depth_anything_v2_vitl.pth` (335.3M params)
- `depth_anything_v2_vitg.pth` (1.3B params, coming soon)
- Metric models: `depth_anything_v2_metric_hypersim_vitl.pth`, etc.

## Architecture

**DepthAnythingV2** (`depth_anything_v2/dpt.py`) — a DINOv2 encoder + DPT decoder head for monocular depth estimation.

- **Encoder** (`depth_anything_v2/dinov2.py`): DINOv2 Vision Transformer (ViT) with 4 variants (small/base/large/giant). Uses intermediate features from 4 specific layers (configurable via `intermediate_layer_idx`). Patch size 14, input size 518.
- **Decoder head** (`DPTHead` in `dpt.py`): Projects the 4 intermediate feature maps to output channels, resizes them to match spatial resolutions, then runs a DPT-style fusion block chain (4 refinement stages with residual conv units) and final output conv layers to produce a 1-channel depth map.
- **`infer_image()`**: Entry point for inference. Converts image to tensor via `image2tensor()` (resize + normalize), runs forward pass, interpolates depth back to original resolution.

**Metric depth** (`metric_depth/`): A separate training pipeline that fine-tunes the same architecture with a `max_depth` parameter. Uses SiLog loss, supports Hypersim/KITTI/VKITTI2 datasets, distributed training via `torch.distributed.launch`.

## Key design details

- `DepthAnythingV2.infer_image(raw_image, input_size)` handles the full inference pipeline: image prep → forward → resize to original dims → return numpy depth map
- The `model_configs` dict in each script maps encoder names to their architectural params (`features`, `out_channels`)
- `use_clstoken=False` in `DPTHead` means the [CLS] token from intermediate layers is not used in the decoder (only patch tokens)
- DINOv2 layers are under `depth_anything_v2/dinov2_layers/` but are generally not modified — they implement standard ViT components (attention, MLP, SwiGLU, patch embedding) with memory-efficient attention
- Metric depth training requires distributed setup (`dist_helper.py`) and uses TensorBoard for logging

## Shared utilities (`depth_anything_v2/util/depth_eval.py`)

Central module for depth evaluation and point cloud generation across all scripts.

**Functions:**
- `valid_mask(pred, gt, max_depth, filter_mono, drop_border)` — boolean mask filtering invalid/non-finite/too-deep pixels; `drop_border` excludes rows 0-18, 466-479, cols 629-639
- `ransac_reciprocal_fit(mono, gt, threshold=0.3, max_trials=500)` — robust fit of `1/gt = a*mono + b` using sklearn RANSACRegressor, returns (a, b, n_inliers)
- `visualize(aligned, gt, diff, save_path, *, sparse, max_depth, gt_label, drop_border)` — 2×3 evaluation figure (depth maps, scatter, histogram, error overlay). `sparse=True` NaN-masks GT/diff for ToF evaluation.
- `back_project_to_pointcloud(depth, rgb, fx, fy, cx, cy, max_depth)` — back-projects depth to (N×3 points, N×3 uint8 colors)
- `write_ply_ascii(path, points, colors)` — write ASCII PLY (same format as ToF .ply files)
- `rgb_guided_residual_interpolation(sparse_u, sparse_v, sparse_residual, rgb, ...)` — RGB-guided joint bilateral interpolation of sparse residuals to dense map

## Stereo vs Mono comparison (`scripts/stereo_vs.py`)

Script that compares DepthAnythingV2 monocular depth predictions against stereo depth ground truth.

- **Data layout**: `data/benchmark/depth/` stores stereo GT `.npy` files; `depth_vit*/benchmark/` stores mono predictions (named as `{image_name}_raw_depth`).
- **Per-image reciprocal alignment**: Fits `1/stereo = a * mono + b` via RANSAC on valid pixels, then applies `mono_aligned = 1 / (a * mono + b)`.
- **Valid mask**: Filters using `valid_mask()` with MAX_DEPTH=0.6m, `filter_mono=True`, `drop_border=True`.
- **Output**: Per-image 2×3 visualization saved to `depth_compare_vit*/benchmark/`.

## ToF vs Mono calibration (`scripts/tof_vs_mono.py`)

Calibrates mono depth scale using ToF point clouds as ground truth.

- **Data**: `data/benchmark/match_data/` — 116 matched sets (ToF .ply + RGB left/right + mono .npy)
- **Camera**: fx=fy=293.756744, cx=324.838501, cy=241.260452, W=640, H=480
- **Extrinsics (cam2tof)**: R and t provided; ToF→camera uses `P_cam = R^T @ (P_tof - t)`
- **Pipeline**: load ToF .ply → transform to camera frame → project to sparse depth map → RANSAC reciprocal fit → align mono → evaluate
- **MAX_DEPTH=0.6m**, `drop_border=True`
- **Results** (vitl, 116 frames): MAE=0.0164, RMSE=0.0349
- **Output**: 2×3 visualizations in `tof_compare_vitl/`

## ToF + Mono depth fusion (`scripts/tof_mono_fusion.py` — experimental)

Attempts to fuse scaled mono depth with ToF sparse point clouds into dense point clouds.

- **Fusion method**: RANSAC reciprocal alignment → residual at ToF pixels → RGB-guided bilateral residual interpolation → corrected dense depth → back-project to 3D
- **Known issue**: Interpolation from ~4500 ToF points to 307K pixels (1.5% density) can cause point cloud streaking artifacts. Pure aligned mono depth may be preferred.
- **Output**: `*_aligned.ply` and `*_fused.ply` in `tof_fusion_vitl/ply/`, diagnostic visualizations in `tof_fusion_vitl/vis/`

## Data matching pipeline

- **`data/match_tof_rgb.py`**: Matches ToF .ply files to nearest RGB frames within 50ms timestamp window
- **`data/dedup_match_data.py`**: SSIM-based near-duplicate frame removal (threshold 0.98), keeps first frame of each group
- **Result**: `data/benchmark/match_data/` with 116 matched sets across `tof/`, `rgb_left/`, `rgb_right/`

## Camera parameters

- Intrinsics: fx=fy=293.756744, cx=324.838501, cy=241.260452, W=640, H=480
- cam2tof extrinsics: R = [[0.004059, -0.001540, 0.999991], [-0.999847, -0.017029, 0.004032], [0.017023, -0.999854, -0.001609]], t = [0.165400, 0.032500, 0.062150]
- Black border regions: rows 0-18, rows 466-479, cols 629-639 (masked via `drop_border`)
