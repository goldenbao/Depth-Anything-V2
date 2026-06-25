"""Deduplicate match_data: keep only the first frame from each near-dup group."""
import os
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

MATCH_DIR = Path(__file__).parent / "benchmark" / "match_data"
SSIM_THRESH = 0.98

def ssim(a, b):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a2 = ((a - mu_a) ** 2).mean()
    sigma_b2 = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    return (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2) / \
           (mu_a ** 2 + mu_b ** 2 + C1) / (sigma_a2 + sigma_b2 + C2)

def extract_ts(filename):
    m = re.search(r"_(\d+)_", filename)
    return int(m.group(1)) if m else None

# --- Find near-duplicate groups in rgb_left ---
img_dir = MATCH_DIR / "rgb_left"
files = sorted(os.listdir(img_dir), key=lambda f: extract_ts(f))

# group consecutive near-duplicates
groups = []
cur = [files[0]]
cache = {}

for i in range(1, len(files)):
    prev_f, cur_f = files[i - 1], files[i]
    for f in (prev_f, cur_f):
        if f not in cache:
            cache[f] = cv2.imread(str(img_dir / f), cv2.IMREAD_GRAYSCALE)
    s = ssim(cache[prev_f], cache[cur_f])
    if s >= SSIM_THRESH:
        cur.append(cur_f)
    else:
        if len(cur) > 1:
            groups.append(cur)
        cur = [cur_f]
if len(cur) > 1:
    groups.append(cur)

# timestamps to keep and delete
keep_ts = set()
delete_ts = set()
for group in groups:
    # keep first, delete rest
    for i, f in enumerate(group):
        ts = extract_ts(f)
        if ts is None:
            continue
        if i == 0:
            keep_ts.add(ts)
        else:
            delete_ts.add(ts)

print(f"Near-duplicate groups: {len(groups)}")
print(f"Frames to delete: {len(delete_ts)}")

# --- Build ToF -> RGB mapping (same logic as match_tof_rgb.py) ---
tof_dir = MATCH_DIR / "tof"
tof_list = []
for f in os.listdir(tof_dir):
    try:
        ts = int(round(float(f.replace(".ply", "")) * 1000))
        tof_list.append((ts, f))
    except ValueError:
        continue

# collect all left timestamps that exist
left_ts_all = set()
for f in os.listdir(MATCH_DIR / "rgb_left"):
    ts = extract_ts(f)
    if ts is not None:
        left_ts_all.add(ts)
left_ts_sorted = sorted(left_ts_all)

import bisect
TOF_MAX_DIFF = 50
tof_to_rgb = {}  # tof_filename -> matched rgb_ts
for tof_ts, tof_file in tof_list:
    idx = bisect.bisect_left(left_ts_sorted, tof_ts)
    cand = []
    if idx > 0:
        cand.append(left_ts_sorted[idx - 1])
    if idx < len(left_ts_sorted):
        cand.append(left_ts_sorted[idx])
    best_diff, best_ts = None, None
    for c in cand:
        d = abs(c - tof_ts)
        if d <= TOF_MAX_DIFF and (best_diff is None or d < best_diff):
            best_diff, best_ts = d, c
    if best_ts is not None:
        tof_to_rgb[tof_file] = best_ts

# reverse mapping: rgb_ts -> tof files
rgb_to_tof = defaultdict(list)
for tof_f, rgb_ts in tof_to_rgb.items():
    rgb_to_tof[rgb_ts].append(tof_f)

# --- Delete: for each deleted left ts, remove corresponding left/right/tof ---
for side in ["rgb_left", "rgb_right", "tof"]:
    dir_path = MATCH_DIR / side
    if not dir_path.exists():
        continue
    deleted = 0
    for f in os.listdir(dir_path):
        if side in ("rgb_left", "rgb_right"):
            ts = extract_ts(f)
            if ts is not None and ts in delete_ts:
                os.remove(dir_path / f)
                deleted += 1
        elif side == "tof":
            # find tof files whose matched RGB ts is in delete_ts
            if f in tof_to_rgb and tof_to_rgb[f] in delete_ts:
                os.remove(dir_path / f)
                deleted += 1
    print(f"  {side}: removed {deleted} files")

# --- Summary ---
for side in ["tof", "rgb_left", "rgb_right"]:
    d = MATCH_DIR / side
    n = len(os.listdir(d)) if d.exists() else 0
    print(f"  {side}: {n} remaining")
