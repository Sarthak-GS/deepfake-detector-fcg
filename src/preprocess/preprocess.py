"""
preprocess.py
-------------
Preprocessing pipeline for deepfake detection (plan.txt Section 3).

Two main steps:
  1. fetch_landmark_bbox: Detect facial landmarks and bounding boxes
     using face-alignment (2D-FAN) — saves per-video .pickle files
  2. crop_main_face: Align and crop faces using mean face template
     — outputs 150×150 face-cropped .avi videos

Usage:
  python -m src.preprocess.preprocess fetch_landmarks \
      --root-dir datasets/ffpp --video-dir videos --fdata-dir frame_data

  python -m src.preprocess.preprocess crop_faces \
      --root-dir datasets/ffpp --fdata-dir frame_data \
      --crop-dir cropped --mean-face misc/20words_mean_face.npy
"""
import os
import cv2
import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, List


# ─────────────────────────────────────────────
# Step 1: Landmark & BBox Extraction
# ─────────────────────────────────────────────

import torch

@torch.inference_mode()
def fetch_landmarks_for_video(org_path, fa, max_res=1920):
    """
    Extract per-frame landmarks & bboxes.
    Returns a list of dicts (or None) matching the fetch_landmark_bbox.py pickle format:
        [{"landmarks": [lm, ...], "bboxes": [bbox_1d, ...]}, None, ...]
    bboxes are stored as 1-D (4,) arrays so get_video_frame_data can reshape them to (2,2).
    """
    cap_org = cv2.VideoCapture(org_path)
    try:
        width  = cap_org.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap_org.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frames = []

        if max(height, width) > max_res:
            scale = max_res / max(height, width)
        else:
            scale = 1

        max_frames = 300 # Prevent OOM for very long/corrupt videos (approx 10s at 30fps)
        while True:
            ret_org, frame_org = cap_org.read()
            if not ret_org or len(frames) >= max_frames:
                break
            try:
                frame_org = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
                # Always resize (same as fetch_landmark_bbox.py)
                frame_org = cv2.resize(frame_org, None, fx=scale, fy=scale)
                frames.append(frame_org)
            except Exception as e:
                print(f"[WARN] Frame read/resize failed: {e}")
                break

        if len(frames) == 0:
            return [None]

        frame_count = len(frames)
        # Default to None per frame (matches fetch_landmark_bbox.py)
        frame_faces = [None for _ in range(frame_count)]

        batch_size   = 10  # process 10 frames at a time
        batch_indices = []
        batch_frames  = []

        for cnt_frame in range(frame_count):
            batch_frames.append(frames[cnt_frame])
            batch_indices.append(cnt_frame)

            if len(batch_frames) == batch_size or (
                cnt_frame == frame_count - 1 and len(batch_frames) > 0
            ):
                results = fa.get_landmarks_from_batch(
                    torch.tensor(np.stack(batch_frames).transpose((0, 3, 1, 2))),
                    return_bboxes=True,
                )

                batch_landmarks = results[0]
                batch_bboxes    = results[2]

                for index, frame_landmarks, frame_bboxes in zip(
                    batch_indices, batch_landmarks, batch_bboxes
                ):
                    if frame_landmarks is not None and len(frame_landmarks) > 0:
                        # shape: (num_faces, 68, 2), unscaled back to original resolution
                        lms    = np.array(frame_landmarks).reshape(-1, 68, 2) / scale
                        bboxes = np.array(frame_bboxes).reshape(-1, 5)

                        # Store bboxes as 1-D (4,) arrays so get_video_frame_data's
                        # reshape to (2,2) works correctly (matching fetch_landmark_bbox.py)
                        mapped_lms    = [lm for lm in lms]
                        mapped_bboxes = [bbox[:-1] / scale for bbox in bboxes]  # shape (4,)

                        frame_faces[index] = {
                            "landmarks": mapped_lms,
                            "bboxes":    mapped_bboxes,
                        }

                batch_frames.clear()
                batch_indices.clear()

        return frame_faces
    finally:
        cap_org.release()


def run_fetch_landmarks(
    root_dir: str,
    video_subdir: str = 'videos',
    fdata_subdir: str = 'frame_data',
    glob_exp: str = '*/*',
    max_res: int = 1920,
):
    """
    Run landmark extraction for all videos matching glob_exp pattern.
    Saves pickle files to <root_dir>/<fdata_subdir>/<glob_exp>/<name>.pickle
    """
    try:
        import face_alignment
    except ImportError:
        raise ImportError("Install face-alignment: pip install face-alignment")

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device='cuda' if _cuda_available() else 'cpu',
    )

    # Find all videos
    pattern = os.path.join(root_dir, glob_exp, video_subdir, '*.avi')
    video_paths = sorted(glob.glob(pattern))
    print(f"Found {len(video_paths)} videos to process")

    for video_path in tqdm(video_paths, desc="Extracting landmarks"):
        # Compute output path
        # e.g. datasets/ffpp/DF/c23/videos/001_003.avi →
        #      datasets/ffpp/frame_data/DF/c23/001_003.pickle
        rel = os.path.relpath(video_path, root_dir)
        parts = rel.split(os.sep)  # ['DF', 'c23', 'videos', '001_003.avi']
        name = os.path.splitext(parts[-1])[0]
        out_dir = os.path.join(root_dir, fdata_subdir, *parts[:-2])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{name}.pickle')

        if os.path.isfile(out_path):
            continue  # skip already processed

        frame_faces = fetch_landmarks_for_video(video_path, fa, max_res)

        with open(out_path, 'wb') as f:
            pickle.dump(frame_faces, f)


# ─────────────────────────────────────────────
# Step 2: Face Cropping & Alignment
# ─────────────────────────────────────────────

class FaceData:
    def __init__(self, _lm, _bbox, _idx):
        self.ema_lm = _lm
        self.ema_bbox = _bbox
        self.lm = [_lm]
        self.bbox = [_bbox]
        self.idx = [_idx]
        self.paddings = 0

    def last_landmark(self): return self.ema_lm
    def last_bbox(self): return self.ema_bbox
    def face_size(self):
        bbox = self.last_bbox()
        return np.linalg.norm(bbox[0] - bbox[1], axis=-1)
    def d_lm(self, landmarks):
        return np.mean(np.linalg.norm(landmarks - self.last_landmark(), axis=-1), axis=1)
    def d_bbox(self, bboxes):
        return np.mean(np.linalg.norm(bboxes - self.last_bbox(), axis=-1), axis=1)
    def pad(self):
        self.paddings += 1
    def add(self, _lm, _bbox, _idx):
        self.ema_lm = self.ema_lm * 0.5 + _lm * 0.5
        self.ema_bbox = self.ema_bbox * 0.5 + _bbox * 0.5
        self.lm.append(_lm)
        self.bbox.append(_bbox)
        self.idx.append(_idx)
        if self.paddings > 0: self.paddings = 0
    def __len__(self): return len(self.lm)

def get_main_face_data(frame_landmarks, frame_bboxes, d_rate, max_paddings):
    # post-process the extracted frame faces.
    # create face identity database to track landmark motion.
    face_dbs = []
    num_frames = len(frame_landmarks)
    for frame_idx, landmarks, bboxes in zip(range(num_frames), frame_landmarks, frame_bboxes):
        if (
            landmarks is None or len(landmarks) == 0 or
            bboxes is None or len(bboxes) == 0
        ):
            for face in face_dbs:
                face.pad()
        else:
            assert len(landmarks) == len(bboxes), "length of landmark and bbox in frame mismatch."
            num_faces = len(landmarks)
            landmarks = np.stack(landmarks)
            bboxes    = np.stack(bboxes)

            matched_indices = {}

            # find and connect with the closest face in the database.
            for db_idx, db_face in enumerate(face_dbs):
                d = db_face.d_bbox(bboxes) + db_face.d_lm(landmarks)
                if np.min(d) > db_face.face_size() * d_rate * 2:
                    continue
                closest_idx = np.argmin(d)
                proximity   = d[closest_idx]
                if (
                    (not closest_idx in matched_indices) or
                    (matched_indices[closest_idx]["d"] > proximity)
                ):
                    matched_indices[closest_idx] = dict(d=proximity, db_idx=db_idx)

            # (hacky!) pad current frame in advance.
            for db_face in face_dbs:
                db_face.pad()

            # finalize and update the database entity.
            for face_idx, save_data in matched_indices.items():
                face_dbs[save_data["db_idx"]].add(landmarks[face_idx], bboxes[face_idx], frame_idx)

            # create new database entity for untracked landmarks.
            for face_idx, landmark, bbox in zip(range(num_faces), landmarks, bboxes):
                if face_idx in matched_indices:
                    continue
                else:
                    face_dbs.append(FaceData(landmark, bbox, frame_idx))

    if not face_dbs:
        return [], [], []
    # report only the most consistent face in the video.
    main_face = sorted(face_dbs, key=lambda x: len(x), reverse=True)[0]
    return main_face.lm, main_face.bbox, main_face.idx

def affine_transform(
    frame,
    bboxes,
    landmarks,
    reference,
    target_size,
    stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,  # avoid black borders on affine warp
    border_value=0,
):
    stable_reference = np.vstack([reference[x] for x in stable_points])
    stable_reference[:, 0] *= (target_size / 256)
    stable_reference[:, 1] *= (target_size / 256)

    transform = cv2.estimateAffinePartial2D(
        np.vstack([landmarks[x] for x in stable_points]),
        stable_reference, method=cv2.LMEDS
    )[0]

    if transform is None:
        return frame, landmarks, bboxes

    transformed_frame = cv2.warpAffine(
        frame, transform,
        dsize=(target_size, target_size),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )
    transformed_landmarks = (
        np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
    )
    transformed_bboxes = (
        np.matmul(bboxes, transform[:, :2].transpose()) + transform[:, 2].transpose()
    )
    return transformed_frame, transformed_landmarks, transformed_bboxes


def crop_driver(img, bboxes, landmarks, size, start_idx, stop_idx):
    center_x, center_y = np.mean(landmarks[start_idx:stop_idx], axis=0)

    if center_y - size < 0:
        center_y = size + 1
    elif (center_y + size) > img.shape[0]:
        center_y = img.shape[0] - size - 1

    if center_x - size < 0:
        center_x = size + 1
    elif (center_x + size) > img.shape[1]:
        center_x = img.shape[1] - size - 1

    uy, by = int(center_y - size), int(center_y + size)
    lx, rx = int(center_x - size), int(center_x + size)
    cutted_img        = np.copy(img[uy:by, lx:rx])
    cutted_landmarks  = np.copy(landmarks) - [lx, uy]
    cutted_bboxes     = np.copy(bboxes)    - [lx, uy]
    return cutted_img, cutted_landmarks, cutted_bboxes


def get_video_frame_data(frame_faces):
    """
    Convert the raw list-of-dicts pickle payload into parallel lists of
    per-frame landmark and bbox arrays, exactly as crop_main_face.py does.
    Handles both 68-point and 98-point (WFLW) landmark models.
    Bboxes are reshaped from 1-D (4,) to (2, 2).
    """
    frame_landmarks = [[] if f is None else f["landmarks"] for f in frame_faces]
    frame_bboxes    = [[] if f is None else f["bboxes"]    for f in frame_faces]

    assert len(frame_landmarks) == len(frame_bboxes), "length of landmark and bbox mismatch."

    # Map 98-pt WFLW → 68-pt FAN if needed
    _98_to_68_mapping = [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
        26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44,
        45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76,
        77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95,
    ]
    frame_landmarks = [
        [(lm[_98_to_68_mapping] if len(lm) == 98 else lm) for lm in landmarks]
        for landmarks in frame_landmarks
    ]

    # Reshape bboxes: 1-D (4,) → (2, 2)
    frame_bboxes = [
        [
            (bbox.reshape((2, 2)) if len(bbox.shape) == 1 else bbox)
            for bbox in bboxes
        ]
        for bboxes in frame_bboxes
    ]

    return frame_landmarks, frame_bboxes


def crop_patch(
    frames,
    landmarks,
    bboxes,
    indices,
    reference,
    window_margin,
    start_idx,
    stop_idx,
    crop_size,
    target_size,
):
    assert len(landmarks) == len(bboxes), "length of landmarks and bboxes mismatch."

    crop_frames    = []
    crop_bboxes    = []
    crop_landmarks = []

    length = len(frames)

    # preprocess for window margin
    _landmarks = [None for _ in range(length)]
    _bboxes    = [None for _ in range(length)]
    for i, idx in enumerate(indices):
        _landmarks[idx] = landmarks[i]
        _bboxes[idx]    = bboxes[i]

    for frame_idx in range(length):
        if frame_idx not in indices:
            crop_frame    = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            crop_landmark = None
            crop_bbox     = None
        else:
            frame  = frames[frame_idx]
            margin = min(window_margin // 2, frame_idx, length - 1 - frame_idx)

            smoothed_landmarks = np.mean(
                [_landmarks[i] for i in range(frame_idx - margin, frame_idx + margin + 1)
                 if _landmarks[i] is not None],
                axis=0,
            )
            smoothed_landmarks += (
                _landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            )

            smoothed_bboxes = np.mean(
                [_bboxes[i] for i in range(frame_idx - margin, frame_idx + margin + 1)
                 if _bboxes[i] is not None],
                axis=0,
            )
            smoothed_bboxes += (
                _bboxes[frame_idx].mean(axis=0) - smoothed_bboxes.mean(axis=0)
            )

            transformed_frame, transformed_landmarks, transformed_bboxes = affine_transform(
                frame, smoothed_bboxes, smoothed_landmarks, reference,
                target_size=target_size,
            )
            crop_frame, crop_landmark, crop_bbox = crop_driver(
                transformed_frame, transformed_bboxes, transformed_landmarks,
                crop_size // 2, start_idx=start_idx, stop_idx=stop_idx,
            )

        assert crop_frame.shape[0] == crop_frame.shape[1] == crop_size, \
            f"crop size mismatch: got {crop_frame.shape}, expected {crop_size}."

        crop_frames.append(crop_frame)
        crop_landmarks.append(crop_landmark)
        crop_bboxes.append(crop_bbox)

    return np.array(crop_frames), crop_landmarks, crop_bboxes


def run_crop_faces(
    root_dir: str,
    video_subdir: str = 'videos',
    fdata_subdir: str = 'frame_data',
    crop_subdir: str = 'cropped',
    glob_exp: str = '*/*',
    crop_w: int = 150,
    crop_h: int = 150,
    mean_face_path: str = 'misc/20words_mean_face.npy',
    d_rate: float = 0.65,
    max_pad_secs: int = 3,
    min_crop_rate: float = 0.9,
):
    """
    Crop and align faces for all videos.
    Uses saved landmark pickles from Step 1.
    Outputs: <root_dir>/<glob_exp>/<crop_subdir>/videos/<name>.avi
    """
    mean_face = np.load(mean_face_path)

    pattern = os.path.join(root_dir, glob_exp, video_subdir, '*.avi')
    video_paths = sorted(glob.glob(pattern))
    print(f"Found {len(video_paths)} videos to crop")

    for video_path in tqdm(video_paths, desc="Cropping faces"):
        rel    = os.path.relpath(video_path, root_dir)
        parts  = rel.split(os.sep)
        name   = os.path.splitext(parts[-1])[0]
        prefix = parts[:-2]   # e.g. ['DF', 'c23']

        # Landmark pickle path
        lm_path = os.path.join(root_dir, fdata_subdir, *prefix, f'{name}.pickle')
        if not os.path.isfile(lm_path):
            continue

        with open(lm_path, 'rb') as f:
            frame_data_raw = pickle.load(f)

        # Decode pickle using the same logic as crop_main_face.py
        try:
            frame_landmarks, frame_bboxes = get_video_frame_data(frame_data_raw)
        except Exception as e:
            print(f"[WARN] Could not decode landmark pickle for {name}: {e}")
            continue

        # Output video
        out_dir = os.path.join(root_dir, *prefix, crop_subdir, 'videos')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{name}.avi')

        if os.path.isfile(out_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        
        fps = round(cap.get(cv2.CAP_PROP_FPS)) or 25  # round like crop_main_face.py
        frames = []
        max_frames = 300  # Avoid OOM
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or len(frames) >= max_frames:
                break
            frames.append(frame.copy())
        cap.release()

        if len(frames) != len(frame_landmarks):
            continue

        landmarks, bboxes, indices = get_main_face_data(
            frame_landmarks=frame_landmarks,
            frame_bboxes=frame_bboxes,
            d_rate=d_rate,
            max_paddings=fps * max_pad_secs,
        )

        if len(landmarks) < len(frames) * min_crop_rate:
            continue

        crop_frames, _, _ = crop_patch(
            frames, landmarks, bboxes, indices, mean_face,
            window_margin=12, start_idx=15, stop_idx=68,
            crop_size=crop_w, target_size=256,
        )

        # Match crop_main_face.py: use FFV1 lossless codec
        fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, crop_h))
        for frame in crop_frames:
            writer.write(frame)
        writer.release()

# Variables for simple preprocess multiprocessing pool
fa_model_simple = None
mean_face_simple = None

def _init_worker_simple(mean_face_path):
    global fa_model_simple
    global mean_face_simple
    mean_face_simple = np.load(mean_face_path)
    import torch
    import face_alignment
    fa_model_simple = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector='sfd',
        dtype=torch.float16,
        flip_input=False,
        device='cuda' if _cuda_available() else 'cpu',
    )

def _process_video_simple(args):
    vid_path, out_path, crop_w, crop_h = args
    # out_path is always forced to .avi by run_simple_preprocess
    if os.path.exists(out_path):
        return True
    try:
        # fetch_landmarks_for_video returns list-of-dicts (or None), matching pickle format
        frame_faces = fetch_landmarks_for_video(vid_path, fa_model_simple, max_res=1920)
        if not any(f is not None for f in frame_faces):
            return False

        # Decode using the same path as run_crop_faces / crop_main_face.py
        frame_landmarks, frame_bboxes = get_video_frame_data(frame_faces)

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return False

        fps = round(cap.get(cv2.CAP_PROP_FPS)) or 25
        frames = []
        max_frames = 400 # Limit to 10s to prevent OOM
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or len(frames) >= max_frames:
                break
            frames.append(frame.copy())
        cap.release()

        if len(frames) != len(frame_landmarks):
            return False

        landmarks, bboxes, indices = get_main_face_data(
            frame_landmarks=frame_landmarks,
            frame_bboxes=frame_bboxes,
            d_rate=1.2,                  # lenient tracking for diverse simple datasets
            max_paddings=int(fps * 10),  # allow up to 10-second gap
        )

        if len(landmarks) < len(frames) * 0.05:  # min_crop_rate = 0.05
            return False

        crop_frames, _, _ = crop_patch(
            frames, landmarks, bboxes, indices, mean_face_simple,
            window_margin=16,  # wider window → smoother temporal landmark smoothing
            start_idx=15, stop_idx=68,
            crop_size=crop_w, target_size=256,
        )

        # out_path is already .avi — write directly, no symlinks
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")  # FFV1 lossless
        writer = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, crop_h))
        for frame in crop_frames:
            writer.write(frame)
        writer.release()

        return True
    except Exception as e:
        print(f"    [WARN] Failed to process {os.path.basename(vid_path)}: {e}")
        return False

def run_simple_preprocess(
    in_dir: str,
    out_dir: str,
    mean_face_path: str = 'misc/20words_mean_face.npy',
    crop_w: int = 224,
    crop_h: int = 224,
):
    """
    Detects faces, extracts landmarks, and crops videos from a simple dataset
    with `fake` and `real` subdirectories. (Uses multiprocessing)

    Output format: single .avi file per video (FFV1 lossless).
    No symlinks are created — one file, one format.
    """
    import multiprocessing as mp

    tasks = []
    for split in ['fake', 'real']:
        src_dir = os.path.join(in_dir, split)
        dst_dir = os.path.join(out_dir, split)

        if not os.path.isdir(src_dir):
            continue

        os.makedirs(dst_dir, exist_ok=True)
        videos = sorted(glob.glob(os.path.join(src_dir, '*.*')))

        for vid_path in videos:
            # Always output as .avi — avoids dual-file issue (.mp4 symlink + .avi)
            stem = os.path.splitext(os.path.basename(vid_path))[0]
            out_path = os.path.join(dst_dir, stem + '.avi')
            tasks.append((vid_path, out_path, crop_w, crop_h))

    if not tasks:
        print("  [Prep Simple] No videos found.")
        return

    print(f"  [Prep Simple] Cropping {len(tasks)} videos using multiprocessing...")

    mp.set_start_method('spawn', force=True)
    num_workers = min(mp.cpu_count(),3)

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker_simple,
        initargs=(mean_face_path,),
    ) as pool:
        results = list(
            tqdm(pool.imap_unordered(_process_video_simple, tasks), total=len(tasks))
        )

    success_count = sum(1 for r in results if r)
    print(f"  [Prep Simple] Finished cropping. Success: {success_count}/{len(tasks)}")



# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def crop_face_from_frame(frame, lm_list, reference, crop_w=224, crop_h=224):
    """
    Crop one face from a single frame given a list of landmarks.
    Used by inference.py for on-the-fly cropping.
    """
    if not lm_list or len(lm_list) == 0:
        return None
        
    landmarks = lm_list[0] # take the first face
    
    if len(landmarks) == 98:
        _98_to_68_mapping = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
            26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44,
            45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76,
            77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
            89, 90, 91, 92, 93, 94, 95,
        ]
        landmarks = landmarks[_98_to_68_mapping]
        
    dummy_bboxes = np.zeros((2, 2))
    
    transformed_frame, transformed_landmarks, _ = affine_transform(
        frame, dummy_bboxes, landmarks, reference,
        target_size=256,
    )
    
    crop_frame, _, _ = crop_driver(
        transformed_frame, dummy_bboxes, transformed_landmarks,
        crop_w // 2, start_idx=15, stop_idx=68,
    )
    
    if crop_frame.shape[0] != crop_h or crop_frame.shape[1] != crop_w:
        crop_frame = cv2.resize(crop_frame, (crop_w, crop_h))
        
    return crop_frame


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFake Preprocessing Tool')
    subparsers = parser.add_subparsers(dest='command')

    # fetch_landmarks subcommand
    p1 = subparsers.add_parser('fetch_landmarks', help='Extract facial landmarks')
    p1.add_argument('--root-dir',   required=True)
    p1.add_argument('--video-dir',  default='videos')
    p1.add_argument('--fdata-dir',  default='frame_data')
    p1.add_argument('--glob-exp',   default='*/*')
    p1.add_argument('--max-res',    type=int, default=1920)

    # crop_faces subcommand
    p2 = subparsers.add_parser('crop_faces', help='Crop and align faces')
    p2.add_argument('--root-dir',     required=True)
    p2.add_argument('--video-dir',    default='videos')
    p2.add_argument('--fdata-dir',    default='frame_data')
    p2.add_argument('--crop-dir',     default='cropped')
    p2.add_argument('--glob-exp',     default='*/*')
    p2.add_argument('--crop-width',   type=int, default=150)
    p2.add_argument('--crop-height',  type=int, default=150)
    p2.add_argument('--mean-face',    default='misc/20words_mean_face.npy')

    args = parser.parse_args()

    if args.command == 'fetch_landmarks':
        run_fetch_landmarks(
            args.root_dir, args.video_dir, args.fdata_dir,
            args.glob_exp, args.max_res
        )
    elif args.command == 'crop_faces':
        run_crop_faces(
            args.root_dir, args.video_dir, args.fdata_dir,
            args.crop_dir, args.glob_exp,
            args.crop_width, args.crop_height, args.mean_face
        )
    else:
        parser.print_help()
