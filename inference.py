"""
inference.py
------------
Single-video inference for the EnhancedDeepfakeDetector (plan.txt Section 8).

Input:  video path (.mp4, .avi, .mov)
Output:
  - overall prediction: REAL / FAKE
  - confidence score: P(fake)
  - frame-level probabilities
  - attention maps per face part (saved as images if requested)

Usage:
  python inference.py \
      --video path/to/video.mp4 \
      --checkpoint checkpoints/run1/best.pt \
      --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
      [--save-heatmaps]
      [--output-dir results/]
"""

import os
import sys
import json
import argparse
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.model import EnhancedDeepfakeDetector
from src.dataset.dataset import to_clip_tensor
from src.preprocess.preprocess import (
    fetch_landmarks_for_video,
    get_video_frame_data,
    get_main_face_data,
    crop_patch,
    _cuda_available,
)

warnings.filterwarnings('ignore')


# Face part names in synoptic order
FACE_PARTS = ['lips', 'skin', 'eyes', 'nose']


def parse_args():
    p = argparse.ArgumentParser(description='Deepfake Video Inference')
    p.add_argument('--video',          required=True, help='Input video path')
    p.add_argument('--checkpoint',     required=True, help='Path to best.pt')
    p.add_argument('--face-features',  required=True,
                   help='Path to L14_real_semantic_patches_v4_2000.pickle')
    p.add_argument('--output-dir',     default='inference_results',
                   help='Save results here')
    p.add_argument('--num-frames',     type=int, default=8)
    p.add_argument('--clip-duration',  type=float, default=3.0)
    p.add_argument('--stride-sec',     type=float, default=1.0,
                   help='Sliding window stride in seconds')
    p.add_argument('--save-heatmaps',  action='store_true',
                   help='Save attention heatmap images')
    p.add_argument('--save-plot',      action='store_true',
                   help='Save frame-level probability and LSTM attention plot')
    p.add_argument('--save-video',     action='store_true',
                   help='Save annotated output video with bounding boxes')
    return p.parse_args()


def load_model(checkpoint_path: str, face_features_path: str,
               num_frames: int, device: torch.device) -> EnhancedDeepfakeDetector:
    model = EnhancedDeepfakeDetector(
        face_feature_path=face_features_path,
        num_frames=num_frames,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


def extract_all_clips_with_bboxes(video_path: str, num_frames: int, clip_duration: float, stride_sec: float = 1.0, fa=None, mean_face=None):
    """
    Extract sliding-window clips using the EXACT same preprocessing logic as training.
    Returns: list of (cropped_frames, start_frame_idx, bboxes, frame_data), fps
    """
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    print(fps)
    
    # 1. Read all frames (limited to 400 for safety, like training)
    frames = []
    max_frames = 400
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        frames.append(frame.copy())
    cap.release()
    total = len(frames)
    
    if total <= 0:
        return [], fps

    print(f"  [Info] Detecting faces in {total} frames... (Training-matched pipeline)")
    
    # 2. Fetch landmarks
    frame_faces = []
    if fa is not None:
        frame_faces = fetch_landmarks_for_video(video_path, fa, max_res=1920)
        
    if not any(f is not None for f in frame_faces):
        print("  [Warning] No faces detected at all.")
        return [], fps
        
    # 3. Decode frame data
    frame_landmarks, frame_bboxes = get_video_frame_data(frame_faces)

    if len(frames) != len(frame_landmarks):
        # Truncate to min
        min_len = min(len(frames), len(frame_landmarks))
        frames = frames[:min_len]
        frame_landmarks = frame_landmarks[:min_len]
        frame_bboxes = frame_bboxes[:min_len]

    # 4. Get Main Face Data with padding and d_rate matches training
    landmarks, bboxes, indices = get_main_face_data(
        frame_landmarks=frame_landmarks,
        frame_bboxes=frame_bboxes,
        d_rate=1.2,
        max_paddings=int(fps * 10),
    )

    if len(landmarks) < len(frames) * 0.05:
        print("  [Warning] Face tracking failed or too few tracked frames.")
        return [], fps

    # 5. Crop Patch (matches training exactly)
    crop_w = 224
    crop_frames_full, crop_landmarks_full, crop_bboxes_full = crop_patch(
        frames, landmarks, bboxes, indices, mean_face,
        window_margin=16,
        start_idx=15, stop_idx=68,
        crop_size=crop_w, target_size=256,
    )

    # 6. Sliding Window to extract 'num_frames' frames every 'clip_duration'
    clip_len = max(int(clip_duration * fps), num_frames)
    stride_len = max(1, int(stride_sec * fps))
    clips = []
    start = 0

    while start < total:
        end = min(start + clip_len, total)
        actual_len = end - start
        
        if actual_len <= num_frames:
            indices_in_clip = list(range(actual_len))
            while len(indices_in_clip) < num_frames:
                indices_in_clip.append(indices_in_clip[-1])
        else:
            step = (actual_len - 1) / (num_frames - 1)
            indices_in_clip = [round(i * step) for i in range(num_frames)]

        clip_frames = []
        clip_bboxes = []
        
        has_valid = False
        for idx in indices_in_clip:
            abs_idx = start + idx
            if abs_idx < len(crop_frames_full):
                frm = crop_frames_full[abs_idx]
                if np.sum(frm) > 0: # Check if frame actually contains face crop data
                    has_valid = True
                clip_frames.append(frm)
                clip_bboxes.append(crop_bboxes_full[abs_idx])
            else:
                clip_frames.append(np.zeros((crop_w, crop_w, 3), dtype=np.uint8))
                clip_bboxes.append(None)
                
        # Only add valid clips
        if has_valid and len(clip_frames) == num_frames:
            # We pass original frame_faces so the annotator can draw full frame bboxes
            clips.append((clip_frames, start, clip_bboxes, frame_faces))

        if start + clip_len >= total:
            break
        start += stride_len

    return clips, fps


def frames_to_tensors(frames, device):
    """Convert list of BGR frames to (x, x_raw) tensors."""
    x_list, raw_list = [], []
    for f in frames:
        x, xr = to_clip_tensor(f)
        x_list.append(x)
        raw_list.append(xr)
    x   = torch.stack(x_list).unsqueeze(0).to(device)    # [1, T, 3, 224, 224]
    raw = torch.stack(raw_list).unsqueeze(0).to(device)
    return x, raw


def save_attention_heatmaps(
    frames,
    attn_maps,   # [L, 1, T, K, 256]
    output_dir: str,
    clip_idx: int,
):
    """Save attention heatmap overlays per face part for a clip."""
    os.makedirs(output_dir, exist_ok=True)
    L, _, T, K, P = attn_maps.shape

    # Average attention across all layers
    avg_attn = attn_maps[:, 0, :, :, :].mean(dim=0)   # [T, K, 256]

    for t in range(min(T, len(frames))):
        frame_bgr  = frames[t]
        frame_rgb  = cv2.cvtColor(
            cv2.resize(frame_bgr, (224, 224)), cv2.COLOR_BGR2RGB
        )

        fig, axes = plt.subplots(1, K + 1, figsize=(5 * (K + 1), 5))
        axes[0].imshow(frame_rgb)
        axes[0].set_title('Original', fontsize=10)
        axes[0].axis('off')

        for k, part in enumerate(FACE_PARTS):
            hm = avg_attn[t, k, :].cpu().numpy().reshape(16, 16)   # 16×16
            hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)

            # Upscale to 224×224
            hm_up = cv2.resize(hm, (224, 224), interpolation=cv2.INTER_LINEAR)

            colored = cm.jet(hm_up)[:, :, :3]  # [224, 224, 3]
            overlay = (frame_rgb / 255.0 * 0.5 + colored * 0.5)
            overlay = np.clip(overlay, 0, 1)

            axes[k + 1].imshow(overlay)
            axes[k + 1].set_title(f'{part}', fontsize=10)
            axes[k + 1].axis('off')

        plt.tight_layout()
        fname = os.path.join(output_dir, f'clip{clip_idx:02d}_frame{t:02d}.png')
        plt.savefig(fname, dpi=80, bbox_inches='tight')
        plt.close()


def save_frame_plot(frame_probs, lstm_attn_list, fps, output_dir: str, video_name: str, clip_duration: float, stride_sec: float = 1.0):
    """Save frame-level P(fake) probability and LSTM attention plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    # x points for clip probabilities (spaced by stride_sec)
    x_probs = np.arange(len(frame_probs)) * stride_sec
    
    # x points for LSTM attention (T per clip)
    # lstm_attn_list is list of [T] arrays
    if lstm_attn_list:
        T = len(lstm_attn_list[0])
        x_attn = np.linspace(0, len(frame_probs) * stride_sec, len(frame_probs) * T)
        attn_flat = np.concatenate(lstm_attn_list)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # Plot Probabilities
    ax1.plot(x_probs, frame_probs, 'b-o', linewidth=2, markersize=6, label='P(fake)')
    ax1.axhline(0.5, color='r', linestyle='--', linewidth=1.5, label='Fake Threshold=0.5')
    ax1.fill_between(x_probs, 0.5, frame_probs,
                     where=np.array(frame_probs) > 0.5,
                     alpha=0.3, color='red', interpolate=True)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('P(Fake)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot LSTM Attention on second y-axis
    if lstm_attn_list:
        ax2 = ax1.twinx()
        ax2.plot(x_attn, attn_flat, 'g-', alpha=0.6, linewidth=1.5, label='LSTM Attention')
        ax2.set_ylabel('Attention Weight', color='g', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(0, max(attn_flat) * 1.5 if max(attn_flat) > 0 else 1.0)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    else:
        ax1.legend(loc='upper right', fontsize=10)

    plt.title(f'Deepfake Probability & Temporal Anomalies — {video_name}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(output_dir, f'{video_name}_analysis_plot.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    return fname

def save_dct_plot(dct_energy: torch.Tensor, output_dir: str, video_name: str, clip_idx: int):
    """Save 8x8 DCT log-energy spectrum heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    
    # dct_energy is [T, 8, 8]. Average over T
    mean_energy = dct_energy.mean(dim=0).cpu().numpy()  # [8, 8]
    
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_energy, cmap='inferno')
    plt.colorbar(label='Log-Energy')
    plt.title(f'DCT Spectrum (Clip {clip_idx})')
    plt.axis('off')
    
    fname = os.path.join(output_dir, f'{video_name}_dct_clip{clip_idx:02d}.png')
    plt.savefig(fname, dpi=80, bbox_inches='tight')
    plt.close()

def create_annotated_video(input_video: str, output_path: str, clip_probs: list, fps: float, clip_duration: float, frame_data: list, stride_sec: float = 1.0):
    """Create annotated video with bounding boxes from face-alignment & live prediction."""
    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        time_sec = frame_idx / fps
        clip_idx = int(time_sec / stride_sec)
        clip_idx = min(clip_idx, len(clip_probs) - 1)
        prob = clip_probs[clip_idx] if clip_probs else 0.5
        
        is_fake = prob >= 0.5
        color = (0, 0, 255) if is_fake else (0, 255, 0)
        label = f"{'FAKE' if is_fake else 'REAL'} ({prob:.1%})"
        
        # Use accurate bboxes from face-alignment if available
        bbox = None
        if frame_idx < len(frame_data) and frame_data[frame_idx] is not None:
            bbox = frame_data[frame_idx].get('bboxes')
        
        drawn = False
        if bbox is not None and len(bbox) > 0:
            b = bbox[0]
            if len(b.shape) == 1 and b.shape[0] == 4:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                drawn = True
            elif len(b.shape) == 2 and b.shape == (2, 2):
                x1, y1 = int(b[0][0]), int(b[0][1])
                x2, y2 = int(b[1][0]), int(b[1][1])
                drawn = True
                
            if drawn:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, max(30, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        if not drawn:
            # Fallback text
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"  [INFO] Annotated video saved: {output_path}")



@torch.no_grad()
def run_video_inference(
    video_path: str,
    model: EnhancedDeepfakeDetector,
    device: torch.device,
    num_frames: int = 8,
    clip_duration: float = 3.0,
    stride_sec: float = 1.0,
    save_heatmaps: bool = False,
    output_dir: str = 'inference_results',
    save_plot: bool = False,
    save_video: bool = False,
) -> dict:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[INFO] Processing: {video_path}")

    # Initialize Face Detector (GPU)
    import face_alignment
    print("  [Info] Initializing high-quality face detector...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device='cuda' if _cuda_available() else 'cpu',
    )
    
    # Load Mean Face for alignment
    mean_face = np.load('misc/20words_mean_face.npy')

    # Extract sliding-window clips and detect faces
    clips, fps = extract_all_clips_with_bboxes(video_path, num_frames, clip_duration, stride_sec=stride_sec, fa=fa, mean_face=mean_face)
    if not clips:
        print("[WARNING] No valid clips extracted. Returning UNKNOWN.")
        return {'prediction': 'UNKNOWN', 'confidence': 0.5, 'clip_probs': []}

    print(f"  Extracted {len(clips)} clips at {fps:.1f} FPS")

    all_fake_probs  = []
    all_spatial     = []
    all_temporal    = []
    all_freq        = []
    lstm_attn_list  = []
    global_frame_data = clips[0][3] if clips else []

    for clip_idx, (frames, start_frame, bboxes, frame_data) in enumerate(clips):
        # 'frames' are now correctly cropped/aligned for the model logic
        x, x_raw = frames_to_tensors(frames, device)

        logits_final, logits_s, logits_t, logits_f, logits_a, sq = model(x, x_raw)

        p_final = F.softmax(logits_final, dim=-1)[0, 1].item()
        p_s     = F.softmax(logits_s, dim=-1)[0, 1].item()
        p_t     = F.softmax(logits_t, dim=-1)[0, 1].item()
        p_f     = F.softmax(logits_f, dim=-1)[0, 1].item()

        all_fake_probs.append(p_final)
        all_spatial.append(p_s)
        all_temporal.append(p_t)
        all_freq.append(p_f)

        # Retrieve Temporal Attention
        t_attn = model.get_temporal_attention()
        if t_attn is not None:
            lstm_attn_list.append(t_attn[0].cpu().numpy())  # [T]

        # Save DCT Spectrum plot if saving plots
        if save_plot:
            dct_env = model.get_dct_spectrum()
            if dct_env is not None:
                # dct_env is [T, 8, 8]
                save_dct_plot(dct_env, os.path.join(output_dir, video_name, 'dct'), video_name, clip_idx)

        # Save attention heatmaps if requested
        if save_heatmaps:
            attn_maps = model.get_attention_maps()
            if attn_maps is not None:
                hm_dir = os.path.join(output_dir, video_name, 'heatmaps')
                save_attention_heatmaps(frames, attn_maps, hm_dir, clip_idx)

    overall_prob = float(np.mean(all_fake_probs))
    prediction   = 'FAKE' if overall_prob >= 0.5 else 'REAL'

    result = {
        'prediction':     prediction,
        'confidence':     round(overall_prob, 4),
        'clip_probs':     [round(p, 4) for p in all_fake_probs],
        'spatial_score':  round(float(np.mean(all_spatial)),  4),
        'temporal_score': round(float(np.mean(all_temporal)), 4),
        'freq_score':     round(float(np.mean(all_freq)),     4),
        'num_clips':      len(clips),
        'video': video_path,
    }

    # Save frame-level plot
    if save_plot:
        plot_path = save_frame_plot(all_fake_probs, lstm_attn_list, fps, output_dir, video_name, clip_duration, stride_sec)
        result['plot_path'] = plot_path
        print(f"  Plot saved: {plot_path}")

    # Generate Annotated Video
    if save_video:
        vid_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
        create_annotated_video(video_path, vid_path, all_fake_probs, fps, clip_duration, global_frame_data, stride_sec)
        result['annotated_video'] = vid_path

    return result


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(
        args.checkpoint, args.face_features, args.num_frames, device
    )

    # Run inference
    result = run_video_inference(
        video_path=args.video,
        model=model,
        device=device,
        num_frames=args.num_frames,
        clip_duration=args.clip_duration,
        stride_sec=args.stride_sec,
        save_heatmaps=args.save_heatmaps,
        output_dir=args.output_dir,
        save_plot=args.save_plot,
        save_video=args.save_video,
    )

    # Print results
    print("\n" + "="*50)
    print(f"  PREDICTION  : {result['prediction']}")
    print(f"  CONFIDENCE  : {result['confidence']:.1%}")
    print(f"  Spatial     : {result['spatial_score']:.4f}")
    print(f"  Temporal    : {result['temporal_score']:.4f}")
    print(f"  Frequency   : {result['freq_score']:.4f}")
    print(f"  Clips       : {result['num_clips']}")
    print("="*50)

    # Save JSON
    out_path = os.path.join(
        args.output_dir,
        os.path.splitext(os.path.basename(args.video))[0] + '_result.json'
    )
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Results saved to {out_path}")


if __name__ == '__main__':
    main()
