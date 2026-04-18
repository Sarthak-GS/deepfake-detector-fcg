"""
dataset.py
----------
Dataset implementation for deepfake detection (plan.txt Section 2, 3).

Supports:
  - FaceForensics++ (FF++) c23 training & validation      → FFPPDataset
  - Celeb-DF v2 (CDF) cross-dataset testing               → CDFDataset
  - Simple flat-folder dataset (fake/ real/ subfolders)   → SimpleVideoDataset
    Used for small-subset pipeline verification runs:
      data_subset_trimmed/fake/*.mp4
      data_subset_trimmed/real/*.mp4

Each item returns:
  x    : [T, 3, 224, 224]  CLIP-normalized frames
  x_raw: [T, 3, 224, 224]  raw uint8 frames [0-255] for DCT branch
  label: int (0=real, 1=fake)
  name : str  video stem
"""

import os
import json
import random
import glob
import cv2
import traceback
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────
# CLIP normalization transform (plan.txt 3.3)
# ─────────────────────────────────────────────

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

clip_normalize_eval = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def to_clip_tensor(frame_bgr: np.ndarray, size: int = 224, is_training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a single BGR numpy frame to:
      x    : [3, 224, 224] float32 CLIP-normalized
      x_raw: [3, 224, 224] float32 [0-255] for DCT

    NOTE: For training, use to_clip_tensor_batch() instead to ensure
    consistent augmentation across all frames in a clip.

    Args:
        frame_bgr: [H, W, 3] uint8 BGR (from OpenCV)
        size     : target resolution (224)
    Returns:
        (x_normalized, x_raw)
    """
    # Resize
    frame = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    # BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Raw float [0-255]
    x_raw = torch.from_numpy(frame).float().permute(2, 0, 1)  # [3, H, W]

    # Normalize (no augmentation — use to_clip_tensor_batch for training)
    x = clip_normalize_eval(frame)

    return x, x_raw


def to_clip_tensor_batch(
    frames_bgr: List[np.ndarray], size: int = 224, is_training: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of BGR frames with CONSISTENT augmentation across all frames.

    Augmentation is decided once per clip and applied identically to every frame,
    preventing the temporal branch from learning augmentation artifacts as "fake".

    Args:
        frames_bgr: list of [H, W, 3] uint8 BGR frames
        size      : target resolution (224)
        is_training: whether to apply augmentation
    Returns:
        (x_batch, x_raw_batch) each [T, 3, 224, 224]
    """
    # Decide augmentation params once for the whole clip
    do_flip = False
    brightness_factor = 1.0
    contrast_factor = 1.0
    saturation_factor = 1.0

    if is_training:
        do_flip = random.random() < 0.5
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)
        saturation_factor = random.uniform(0.9, 1.1)

    x_list, raw_list = [], []
    for frame_bgr in frames_bgr:
        frame = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if is_training:
            if do_flip:
                frame = np.fliplr(frame).copy()
            # Apply brightness/contrast/saturation consistently
            frame_f = frame.astype(np.float32)
            # Brightness
            frame_f = frame_f * brightness_factor
            # Contrast (around mean)
            mean_val = frame_f.mean()
            frame_f = (frame_f - mean_val) * contrast_factor + mean_val
            frame_f = np.clip(frame_f, 0, 255).astype(np.uint8)
            # Saturation (via HSV)
            hsv = cv2.cvtColor(frame_f, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            frame_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            frame = frame_f

        # Raw float [0-255] — augmentation is applied to raw too for consistency
        x_raw = torch.from_numpy(np.ascontiguousarray(frame)).float().permute(2, 0, 1)

        # CLIP normalize
        x = clip_normalize_eval(np.ascontiguousarray(frame))

        x_list.append(x)
        raw_list.append(x_raw)

    return torch.stack(x_list), torch.stack(raw_list)


def load_video_frames(
    video_path: str,
    num_frames: int = 8,
    clip_duration: float = 3.0,
    random_sample: bool = True,
) -> Tuple[Optional[List[np.ndarray]], float]:
    """
    Sample num_frames uniformly from a video clip.
    Safely bypasses OpenCV CAP_PROP_FRAME_COUNT bugs (which often returns 0 for FFV1/AVI).
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 0.0

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Read frames sequentially into memory because FFV1 frame seeking is unreliable
        # and CAP_PROP_FRAME_COUNT is often artificially 0.
        all_frames = []
        max_frames = 900 # Absolute max ~30s at 30fps to prevent OOM
        while cap.isOpened() and len(all_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        total = len(all_frames)
        if total <= 0:
            return None, fps

        clip_frames = int(clip_duration * fps)
        clip_frames = min(clip_frames, total)

        if clip_frames <= 0:
            return None, fps

        # Choose clip start
        if random_sample and total > clip_frames:
            start = random.randint(0, total - clip_frames)
        else:
            start = 0

        # Determine target indices
        indices = []
        if clip_frames <= num_frames:
            indices = list(range(start, start + clip_frames))
            while len(indices) < num_frames:
                indices.append(indices[-1])
        else:
            step = (clip_frames - 1) / (num_frames - 1)
            indices = [round(start + i * step) for i in range(num_frames)]

        # Extract exactly num_frames
        frames = []
        for idx in indices:
            idx = min(idx, total - 1) # Safety clamp
            frames.append(all_frames[idx].copy())
            
        return frames, fps

    except Exception as e:
        print(f"  [ERROR] load_video_frames failed on {video_path}: {e}")
        return None, 0.0


# ─────────────────────────────────────────────
# FFPP Dataset (plan.txt Section 2.2, 2.3)
# ─────────────────────────────────────────────

class FFPPDataset(Dataset):
    """
    FaceForensics++ dataset loader.

    Directory structure expected (from plan.txt Section 2.3):
      datasets/ffpp/real/c23/cropped/videos/*.avi
      datasets/ffpp/DF/c23/cropped/videos/*.avi
      datasets/ffpp/FS/c23/cropped/videos/*.avi
      datasets/ffpp/F2F/c23/cropped/videos/*.avi
      datasets/ffpp/NT/c23/cropped/videos/*.avi
      datasets/ffpp/csv_files/train.json  [[id1, id2], ...]
      datasets/ffpp/csv_files/val.json
      datasets/ffpp/csv_files/test.json

    Sampling strategy: FORCE_PAIR (plan.txt Section 3.6)
      Each item returns a (real, fake) pair from the same identity.
      The collate function interleaves them into a flat batch.
    """

    FAKE_TYPES = ['DF', 'FS', 'F2F', 'NT', 'FSh', 'DFD']

    def __init__(
        self,
        root_dir: str,          # e.g. "datasets/ffpp"
        split: str,             # 'train', 'val', 'test'
        num_frames: int = 8,
        clip_duration: float = 3.0,
        compression: str = 'c23',
        is_training: bool = True,
        limit_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir     = root_dir
        self.split        = split
        self.num_frames   = num_frames
        self.clip_duration = clip_duration
        self.compression  = compression
        self.is_training  = is_training

        # Load identity pairs from JSON
        json_path = os.path.join(root_dir, 'csv_files', f'{split}.json')
        with open(json_path, 'r') as f:
            self.pairs = json.load(f)  # [[id1, id2], ...]

        # Build list of (real_path, fake_path, fake_type) tuples
        self.items = self._build_items()

        if limit_samples is not None:
            random.shuffle(self.items) # Shuffle so we get diverse fake types/identities if limiting
            self.items = self.items[:limit_samples]

    def _build_items(self) -> List[Tuple[str, str, str]]:
        """Build (real_path, fake_path, fake_type) per identity pair."""
        items = []
        cropped_subdir = os.path.join('c23', 'cropped', 'videos') \
                         if self.compression == 'c23' else \
                         os.path.join(self.compression, 'cropped', 'videos')

        real_dir = os.path.join(self.root_dir, 'real', cropped_subdir)

        for fake_type in self.FAKE_TYPES:
            fake_dir = os.path.join(self.root_dir, fake_type, cropped_subdir)
            if not os.path.isdir(fake_dir):
                continue

            for pair in self.pairs:
                id1, id2 = str(pair[0]), str(pair[1])
                # Real video name (FF++ naming: e.g. 000.avi)
                real_name = f"{id1.zfill(3)}.avi"
                # Fake video name (e.g. 000_001.avi)
                fake_name = f"{id1.zfill(3)}_{id2.zfill(3)}.avi"

                real_path = os.path.join(real_dir, real_name)
                fake_path = os.path.join(fake_dir, fake_name)

                if os.path.isfile(real_path) and os.path.isfile(fake_path):
                    items.append((real_path, fake_path, fake_type))

        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        max_retries = 5
        for attempt in range(max_retries):
            real_path, fake_path, fake_type = self.items[idx]
            try:
                # Sample frames from real and fake
                real_frames, fps = load_video_frames(
                    real_path, self.num_frames, self.clip_duration,
                    random_sample=self.is_training
                )
                fake_frames, _   = load_video_frames(
                    fake_path, self.num_frames, self.clip_duration,
                    random_sample=self.is_training
                )

                if real_frames is None or fake_frames is None:
                    print(f"  [WARN] Could not load videos (real or fake). Retrying.")
                    idx = random.randint(0, len(self.items) - 1)
                    continue

                # Convert to tensors with consistent augmentation across all frames
                real_x, real_raw = to_clip_tensor_batch(real_frames, is_training=self.is_training)
                fake_x, fake_raw = to_clip_tensor_batch(fake_frames, is_training=self.is_training)

                return {
                    'real_x':   real_x,
                    'real_raw': real_raw,
                    'fake_x':   fake_x,
                    'fake_raw': fake_raw,
                    'real_name': os.path.basename(real_path),
                    'fake_name': os.path.basename(fake_path),
                    'fake_type': fake_type,
                }
            except Exception as e:
                print(f"  [ERROR] FFPPDataset.__getitem__ crashed on idx={idx}: {e}")
                idx = random.randint(0, len(self.items) - 1)
        
        # Fallback if everything fails
        black = torch.zeros(self.num_frames, 3, 224, 224)
        return {'real_x': black, 'real_raw': black, 'fake_x': black, 'fake_raw': black, 
                'real_name': 'error', 'fake_name': 'error', 'fake_type': 'error'}


def ffpp_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate FORCE_PAIR samples into flat batch.
    Interleaves real and fake: [real0, fake0, real1, fake1, ...]
    Labels: [0, 1, 0, 1, ...]
    """
    x_list, raw_list, labels, names = [], [], [], []

    for item in batch:
        x_list.append(item['real_x'])
        raw_list.append(item['real_raw'])
        labels.append(0)
        names.append(item['real_name'])

        x_list.append(item['fake_x'])
        raw_list.append(item['fake_raw'])
        labels.append(1)
        names.append(item['fake_name'])

    return {
        'x':      torch.stack(x_list),            # [2B, T, 3, 224, 224]
        'x_raw':  torch.stack(raw_list),           # [2B, T, 3, 224, 224]
        'labels': torch.tensor(labels, dtype=torch.long),   # [2B]
        'names':  names,
    }


# ─────────────────────────────────────────────
# CDF Dataset (for cross-dataset test)
# ─────────────────────────────────────────────

class CDFDataset(Dataset):
    """
    Celeb-DF v2 dataset for cross-dataset evaluation.

    Directory structure:
      datasets/cdf/REAL/cropped/videos/*.avi
      datasets/cdf/FAKE/cropped/videos/*.avi
      datasets/cdf/csv_files/test_real.csv   (one path per line)
      datasets/cdf/csv_files/test_fake.csv
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 8,
        clip_duration: float = 3.0,
    ):
        self.root_dir      = root_dir
        self.num_frames    = num_frames
        self.clip_duration = clip_duration

        self.samples = []  # list of (path, label)

        for csv_name, label in [('test_real.csv', 0), ('test_fake.csv', 1)]:
            csv_path = os.path.join(root_dir, 'csv_files', csv_name)
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path, 'r') as f:
                for line in f:
                    name = line.strip()
                    if not name:
                        continue
                    subdir = 'REAL' if label == 0 else 'FAKE'
                    full   = os.path.join(root_dir, subdir, 'cropped', 'videos', name)
                    if os.path.isfile(full):
                        self.samples.append((full, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.samples[idx]
        frames, fps = load_video_frames(
            path, self.num_frames, self.clip_duration, random_sample=False
        )
        if frames is None:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames

        x_list, raw_list = [], []
        for f in frames:
            x, xr = to_clip_tensor(f, is_training=False)
            x_list.append(x)
            raw_list.append(xr)

        return {
            'x':     torch.stack(x_list),
            'x_raw': torch.stack(raw_list),
            'label': torch.tensor(label, dtype=torch.long),
            'name':  os.path.basename(path),
        }


def cdf_collate_fn(batch: List[Dict]) -> Dict:
    return {
        'x':      torch.stack([b['x']     for b in batch]),
        'x_raw':  torch.stack([b['x_raw'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
        'names':  [b['name'] for b in batch],
    }


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def build_ffpp_loaders(
    ffpp_root: str,
    batch_size: int = 3,
    num_frames: int = 8,
    num_workers: int = 4,
    limit_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for FF++.
    Note: batch_size here is number of PAIRS, actual batch size = batch_size * 2.
    """
    train_ds = FFPPDataset(ffpp_root, 'train', num_frames, is_training=True, limit_samples=limit_samples)
    val_ds   = FFPPDataset(ffpp_root, 'val',   num_frames, is_training=False, limit_samples=limit_samples)
    test_ds  = FFPPDataset(ffpp_root, 'test',  num_frames, is_training=False, limit_samples=limit_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=ffpp_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=ffpp_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=ffpp_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_cdf_loader(
    cdf_root: str,
    batch_size: int = 6,
    num_frames: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    ds = CDFDataset(cdf_root, num_frames)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=cdf_collate_fn,
        pin_memory=True,
    )


# ─────────────────────────────────────────────
# SimpleVideoDataset — flat fake/ real/ folders
# (for pipeline verification with small subset)
# ─────────────────────────────────────────────

VIDEO_EXTS = ('*.mp4', '*.avi', '*.mov', '*.mkv')


class SimpleVideoDataset(Dataset):
    """
    Flat-folder dataset: reads from <root>/fake/ and <root>/real/.
    No JSON files, no paired sampling — just independent videos with labels.

    Directory structure:
      <root>/fake/*.mp4   (label=1)
      <root>/real/*.mp4   (label=0)

    Works with data_subset_trimmed/ produced by create_subset.py + trim_videos.py.
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 8,
        clip_duration: float = 3.0,
        is_training: bool = True,
        split: str = 'train',       # 'train', 'val', 'test'
        val_frac: float = 0.15,
        test_frac: float = 0.10,
        seed: int = 42,
        limit_samples: Optional[int] = None,
    ):
        super().__init__()
        self.num_frames    = num_frames
        self.clip_duration = clip_duration
        self.is_training   = is_training

        # Collect all videos
        fake_dir = os.path.join(root_dir, 'fake')
        real_dir = os.path.join(root_dir, 'real')

        fake_vids = self._collect(fake_dir)
        real_vids = self._collect(real_dir)

        if not fake_vids:
            raise FileNotFoundError(f"No fake videos found in: {fake_dir}")
        if not real_vids:
            raise FileNotFoundError(f"No real videos found in: {real_dir}")

        print(f"[SimpleVideoDataset] Found {len(fake_vids)} fake, {len(real_vids)} real")

        # Label each video
        all_samples: List[Tuple[str, int]] = (
            [(v, 1) for v in fake_vids] +
            [(v, 0) for v in real_vids]
        )

        # Deterministic shuffle before splitting
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        n = len(all_samples)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        n_train = n - n_val - n_test

        if split == 'train':
            self.samples = all_samples[:n_train]
        elif split == 'val':
            self.samples = all_samples[n_train:n_train + n_val]
        else:  # test
            self.samples = all_samples[n_train + n_val:]

        if limit_samples is not None:
            self.samples = self.samples[:limit_samples]


        # Log distribution
        n_fake = sum(1 for _, l in self.samples if l == 1)
        n_real = sum(1 for _, l in self.samples if l == 0)
        print(f"  [{split}] {len(self.samples)} samples  "
              f"(fake={n_fake}, real={n_real})")

    @staticmethod
    def _collect(folder: str) -> List[str]:
        """Collect all video files from a folder."""
        if not os.path.isdir(folder):
            return []
        vids = []
        for ext in VIDEO_EXTS:
            vids.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(set(vids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        max_retries = 5
        for attempt in range(max_retries):
            path, label = self.samples[idx]
            try:
                frames, _ = load_video_frames(
                    path, self.num_frames, self.clip_duration,
                    random_sample=self.is_training,
                )

                if frames is None or len(frames) == 0:
                    print(f"  [WARN] Could not load frames from: {path} — retrying different sample.")
                    idx = random.randint(0, len(self.samples) - 1)
                    continue

                # Safety: ensure exactly num_frames
                while len(frames) < self.num_frames:
                    frames.append(frames[-1].copy() if frames else
                                  np.zeros((224, 224, 3), dtype=np.uint8))
                frames = frames[:self.num_frames]

                x, x_raw = to_clip_tensor_batch(frames, is_training=self.is_training)

                return {
                    'x':     x,                                       # [T, 3, 224, 224]
                    'x_raw': x_raw,                                   # [T, 3, 224, 224]
                    'label': torch.tensor(label, dtype=torch.long),
                    'name':  os.path.basename(path),
                }
            except Exception as e:
                print(f"  [ERROR] SimpleVideoDataset.__getitem__ crashed on "
                      f"{os.path.basename(path)} (idx={idx}): {e}")
                idx = random.randint(0, len(self.samples) - 1)

        # Ultimate fallback if everything fails
        print(f"  [ERROR] Exceeded max retries in SimpleVideoDataset, falling back.")
        black = torch.zeros(self.num_frames, 3, 224, 224)
        return {'x': black, 'x_raw': black, 'label': torch.tensor(1, dtype=torch.long), 'name': 'error'}


def simple_collate_fn(batch: List[Dict]) -> Dict:
    """Collate for SimpleVideoDataset — straightforward stack."""
    return {
        'x':      torch.stack([b['x']     for b in batch]),       # [B, T, 3, 224, 224]
        'x_raw':  torch.stack([b['x_raw'] for b in batch]),       # [B, T, 3, 224, 224]
        'labels': torch.stack([b['label'] for b in batch]),        # [B]
        'names':  [b['name'] for b in batch],
    }


def build_simple_loaders(
    root_dir: str,
    batch_size: int = 4,
    num_frames: int = 8,
    num_workers: int = 2,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
    limit_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from a simple flat-folder dataset.

    Args:
        root_dir: path to folder containing fake/ and real/ subdirs
    """
    common = dict(
        root_dir=root_dir, num_frames=num_frames,
        val_frac=val_frac, test_frac=test_frac,
        seed=seed, limit_samples=limit_samples,
    )
    train_ds = SimpleVideoDataset(**common, split='train', is_training=True)
    val_ds   = SimpleVideoDataset(**common, split='val',   is_training=False)
    test_ds  = SimpleVideoDataset(**common, split='test',  is_training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=simple_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=simple_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=simple_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
