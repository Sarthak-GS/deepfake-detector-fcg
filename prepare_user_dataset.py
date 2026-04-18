"""
prepare_user_dataset.py
-----------------------
Maps your actual video dataset (10 real + 30 fake across 6 methods)
into the ffpp directory structure expected by the training pipeline.

Source layout:
  datasets/original/*.mp4           (10 real videos)
  datasets/Deepfakes/*.mp4          (5 fake videos)
  datasets/Face2Face/*.mp4          (5 fake videos)
  datasets/FaceSwap/*.mp4           (5 fake videos)
  datasets/FaceShifter/*.mp4        (5 fake videos)
  datasets/NeuralTextures/*.mp4     (5 fake videos)
  datasets/DeepFakeDetection/*.mp4  (5 fake videos)

Target layout:
  datasets/ffpp/real/c23/cropped/videos/{id}.avi
  datasets/ffpp/{type}/c23/cropped/videos/{id1}_{id2}.avi
  datasets/ffpp/csv_files/train.json, val.json, test.json
"""

import os
import glob
import shutil
import json
import random

random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────
ROOT = "datasets"
FFPP = os.path.join(ROOT, "ffpp")

# Mapping: source folder → ffpp fake type name
FAKE_FOLDERS = {
    "Deepfakes":        "DF",
    "Face2Face":        "F2F",
    "FaceSwap":         "FS",
    "FaceShifter":      "FSh",
    "NeuralTextures":   "NT",
    "DeepFakeDetection":"DFD",
}

# ─── Step 1: Clean old ffpp data ──────────────────────────────────────
for subdir in ["real", "csv_files"] + list(FAKE_FOLDERS.values()):
    d = os.path.join(FFPP, subdir)
    if os.path.isdir(d):
        shutil.rmtree(d)

# ─── Step 2: Copy real videos ─────────────────────────────────────────
real_src = sorted(glob.glob(os.path.join(ROOT, "original", "*.mp4")))
real_out = os.path.join(FFPP, "real", "c23", "cropped", "videos")
os.makedirs(real_out, exist_ok=True)

real_id_map = {}   # original filename stem → assigned numeric ID
for i, src in enumerate(real_src):
    stem = os.path.splitext(os.path.basename(src))[0]
    real_id = f"{i:03d}"
    real_id_map[stem] = real_id
    dst = os.path.join(real_out, f"{real_id}.avi")
    shutil.copy(src, dst)

print(f"Copied {len(real_src)} real videos → {real_out}")

# ─── Step 3: Copy fake videos ────────────────────────────────────────
all_pairs = []

for src_folder, ffpp_type in FAKE_FOLDERS.items():
    fake_src = sorted(glob.glob(os.path.join(ROOT, src_folder, "*.mp4")))
    if not fake_src:
        print(f"  [SKIP] No videos in {src_folder}/")
        continue

    fake_out = os.path.join(FFPP, ffpp_type, "c23", "cropped", "videos")
    os.makedirs(fake_out, exist_ok=True)

    for j, fake_path in enumerate(fake_src):
        # Pair each fake with a real video (round-robin)
        real_idx = j % len(real_src)
        real_id = f"{real_idx:03d}"
        fake_id = f"{j:03d}"

        fake_name = f"{real_id}_{fake_id}.avi"
        dst = os.path.join(fake_out, fake_name)
        shutil.copy(fake_path, dst)

        pair = [real_id, fake_id]
        if pair not in all_pairs:
            all_pairs.append(pair)

    print(f"  {src_folder} → {ffpp_type}: {len(fake_src)} videos")

# ─── Step 4: Create train/val/test splits ─────────────────────────────
random.shuffle(all_pairs)
n = len(all_pairs)

# With 30 fakes: ~20 train, ~5 val, ~5 test
n_train = max(1, int(n * 0.6))
n_val   = max(1, int(n * 0.2))

train_pairs = all_pairs[:n_train]
val_pairs   = all_pairs[n_train:n_train + n_val]
test_pairs  = all_pairs[n_train + n_val:]

# Ensure test has at least 1
if not test_pairs:
    test_pairs = [val_pairs.pop()]

csv_dir = os.path.join(FFPP, "csv_files")
os.makedirs(csv_dir, exist_ok=True)

for name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
    with open(os.path.join(csv_dir, f"{name}.json"), 'w') as f:
        json.dump(pairs, f, indent=2)

print(f"\nSplit: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
print(f"Total pairs: {len(all_pairs)}")
print("\nDataset prepared! Now run:")
# print("  python trim_videos.py")
print("  python train.py --ffpp-root datasets/ffpp --cdf-root datasets/cdf --face-features misc/L14_real_semantic_patches_v4_2000.pickle --output-dir checkpoints/run1 --batch-size 1 --epochs 30 --num-frames 4 --num-workers 0")
