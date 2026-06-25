"""Match ToF point clouds with nearest left/right RGB images (within 50ms)."""
import os
import re
import shutil
import bisect
from pathlib import Path

BASE = Path(__file__).parent / "benchmark"
TOF_DIR = BASE / "tof"
RGB_DIR = BASE / "rgb"
OUT_DIR = BASE / "match_data"

MAX_DIFF_MS = 50

# --- Collect ToF timestamps ---
tof_list = []  # (ts, filename)
for f in os.listdir(TOF_DIR):
    try:
        ts = int(round(float(f.replace(".ply", "")) * 1000))
        tof_list.append((ts, f))
    except ValueError:
        continue
tof_list.sort()

# --- Collect RGB left/right images by timestamp ---
left_by_ts = {}   # ts -> (subdir, filename)
right_by_ts = {}  # ts -> (subdir, filename)
for subdir in os.listdir(RGB_DIR):
    subpath = RGB_DIR / subdir
    if not subpath.is_dir():
        continue
    for f in os.listdir(subpath):
        m = re.match(r"^SLAM_SLAM_L_TX0_(\d+)_640X480\.jpg$", f)
        if m:
            left_by_ts[int(m.group(1))] = (subdir, f)
            continue
        m = re.match(r"^SLAM_SLAM_R_TX0_(\d+)_640X480\.jpg$", f)
        if m:
            right_by_ts[int(m.group(1))] = (subdir, f)

# sorted list of RGB timestamps for nearest-neighbor search
rgb_ts_sorted = sorted(left_by_ts.keys())

print(f"ToF files: {len(tof_list)}")
print(f"Left images: {len(left_by_ts)}")
print(f"Right images: {len(right_by_ts)}")
print(f"Max allowed diff: {MAX_DIFF_MS}ms\n")

# --- For each ToF, find nearest left+right within threshold ---
matched = []  # (tof_ts, tof_file, left_ts, left_subdir, left_file, right_subdir, right_file, diff_ms)

for tof_ts, tof_file in tof_list:
    idx = bisect.bisect_left(rgb_ts_sorted, tof_ts)
    candidates = []
    if idx > 0:
        candidates.append(rgb_ts_sorted[idx - 1])
    if idx < len(rgb_ts_sorted):
        candidates.append(rgb_ts_sorted[idx])

    best_diff = None
    best_ts = None
    for cand_ts in candidates:
        diff = abs(cand_ts - tof_ts)
        if diff <= MAX_DIFF_MS and (best_diff is None or diff < best_diff):
            # check that right image also exists at this timestamp
            if cand_ts in right_by_ts:
                best_diff = diff
                best_ts = cand_ts

    if best_ts is not None:
        left_subdir, left_file = left_by_ts[best_ts]
        right_subdir, right_file = right_by_ts[best_ts]
        matched.append((tof_ts, tof_file, best_ts, left_subdir, left_file,
                        right_subdir, right_file, best_diff))

# --- Deduplicate: keep only the closest ToF per RGB timestamp ---
best_per_rgb = {}  # rgb_ts -> (tof_ts, tof_file, ...)
for entry in matched:
    rgb_ts = entry[2]
    if rgb_ts not in best_per_rgb or entry[7] < best_per_rgb[rgb_ts][7]:
        best_per_rgb[rgb_ts] = entry

matched = sorted(best_per_rgb.values(), key=lambda x: x[0])
print(f"Matched sets (unique RGB): {len(matched)}")

if matched:
    print(f"\nFirst 5 matches:")
    for tof_ts, tof_f, rgb_ts, _, lf, _, rf, diff in matched[:5]:
        print(f"  ToF {tof_f} ({tof_ts}) <-> RGB {lf}/{rf} ({rgb_ts}), diff={diff}ms")
    if len(matched) > 5:
        print(f"  ... and {len(matched) - 5} more")

# --- Copy to match_data ---
tof_out = OUT_DIR / "tof"
left_out = OUT_DIR / "rgb_left"
right_out = OUT_DIR / "rgb_right"
for d in [tof_out, left_out, right_out]:
    d.mkdir(parents=True, exist_ok=True)

# clear previous content
for d in [tof_out, left_out, right_out]:
    for f in os.listdir(d):
        (d / f).unlink()

for tof_ts, tof_file, rgb_ts, left_subdir, left_file, right_subdir, right_file, diff in matched:
    shutil.copy2(TOF_DIR / tof_file, tof_out / tof_file)
    shutil.copy2(RGB_DIR / left_subdir / left_file, left_out / left_file)
    shutil.copy2(RGB_DIR / right_subdir / right_file, right_out / right_file)

print(f"\nCopied to {OUT_DIR}/:")
print(f"  tof/       : {len(os.listdir(tof_out))} files")
print(f"  rgb_left/  : {len(os.listdir(left_out))} files")
print(f"  rgb_right/ : {len(os.listdir(right_out))} files")
