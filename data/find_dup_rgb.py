"""Detect duplicate/near-duplicate RGB images in match_data."""
import os
import re
import hashlib
import cv2
import numpy as np
from pathlib import Path

MATCH_DIR = Path(__file__).parent / "benchmark" / "match_data"
SSIM_THRESH = 0.98  # SSIM >= this → near-duplicate

def ssim(a, b):
    """Structural Similarity Index between two grayscale images."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a2 = ((a - mu_a) ** 2).mean()
    sigma_b2 = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    return (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2) / \
           (mu_a ** 2 + mu_b ** 2 + C1) / (sigma_a2 + sigma_b2 + C2)


for side in ["rgb_left", "rgb_right"]:
    img_dir = MATCH_DIR / side
    if not img_dir.exists():
        continue

    files = sorted(os.listdir(img_dir), key=lambda f: int(re.search(r"_(\d+)_", f).group(1)))

    # --- 1. Exact duplicates ---
    hash_to_files = {}
    for f in files:
        md5 = hashlib.md5(open(img_dir / f, "rb").read()).hexdigest()
        hash_to_files.setdefault(md5, []).append(f)

    print(f"\n{'='*60}")
    print(f"[{side}]  Total: {len(files)}")
    exact_dup_groups = [v for v in hash_to_files.values() if len(v) > 1]
    print(f"Exact duplicates (MD5): {len(exact_dup_groups)} groups")
    for group in exact_dup_groups:
        for f in group:
            sz = os.path.getsize(img_dir / f)
            print(f"    {f}  ({sz} bytes)")

    # --- 2. Near-duplicates by SSIM (consecutive frames) ---
    print(f"\nNear-duplicates (SSIM >= {SSIM_THRESH}, consecutive):")
    imgs = {}
    near_dup_groups = []
    cur_group = [files[0]]

    for i in range(1, len(files)):
        # load only when needed, cache
        prev_f = files[i - 1]
        cur_f = files[i]
        if prev_f not in imgs:
            imgs[prev_f] = cv2.imread(str(img_dir / prev_f), cv2.IMREAD_GRAYSCALE)
        if cur_f not in imgs:
            imgs[cur_f] = cv2.imread(str(img_dir / cur_f), cv2.IMREAD_GRAYSCALE)

        s = ssim(imgs[prev_f], imgs[cur_f])
        if s >= SSIM_THRESH:
            cur_group.append(cur_f)
        else:
            if len(cur_group) > 1:
                near_dup_groups.append(cur_group)
            cur_group = [cur_f]

    if len(cur_group) > 1:
        near_dup_groups.append(cur_group)

    print(f"  {len(near_dup_groups)} groups:")
    for group in near_dup_groups:
        ts = [re.search(r"_(\d+)_", f).group(1) for f in group]
        print(f"    [{ts[0]}..{ts[-1]}]  ({len(group)} frames)  {group[0]} ... {group[-1]}")
