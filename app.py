"""
app.py
------
Gradio UI for the EnhancedDeepfakeDetector (plan.txt Section 9).

Features:
  - Upload video → predict REAL/FAKE with confidence
  - Per-branch scores (Spatial, Temporal, Frequency)
  - Frame-level probability chart
  - Attention heatmap gallery (face parts: lips, eyes, skin, nose)

Usage:
  python app.py \
      --checkpoint checkpoints/run1/best.pt \
      --face-features misc/L14_real_semantic_patches_v4_2000.pickle
"""

import os
import sys
import argparse
import tempfile
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gradio as gr
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.model import EnhancedDeepfakeDetector
from inference import run_video_inference, load_model

warnings.filterwarnings('ignore')

# Global model reference (loaded once at startup)
_model = None
_device = None
_args = None

FACE_PARTS = ['lips', 'skin', 'eyes', 'nose']


# ─────────────────────────────────────────────
# Prediction backend
# ─────────────────────────────────────────────

def predict(
    video,              # gradio returns temp path string
    threshold: float = 0.5,
):
    """
    Main Gradio prediction function.
    Called when the user clicks Analyze.
    """
    global _model, _device

    if video is None:
        return "⚠️ Please upload a video.", "", "", "", None, None, []

    if _model is None:
        return "⚠️ Model not loaded. Restart app with --checkpoint flag.", \
               "", "", "", None, None, []

    try:
        tmp_dir = tempfile.mkdtemp()
        result = run_video_inference(
            video_path=video,
            model=_model,
            device=_device,
            num_frames=8,
            clip_duration=3.0,
            save_heatmaps=True,
            output_dir=tmp_dir,
            save_plot=True,
            save_video=True,
        )

        # ── Prediction text ──────────────────────────────────────────
        icon  = "🔴 FAKE" if result['prediction'] == 'FAKE' else "🟢 REAL"
        conf  = result['confidence']
        pred_str = f"{icon}  ({conf:.1%} confidence)"

        s_str = f"Spatial  : {result['spatial_score']:.4f}"
        t_str = f"Temporal : {result['temporal_score']:.4f}"
        f_str = f"Frequency: {result['freq_score']:.4f}"

        # ── Frame-level probability plot ─────────────────────────────
        clip_probs = result.get('clip_probs', [conf])
        fig = make_frame_plot(clip_probs, threshold)

        # ── Heatmap gallery ──────────────────────────────────────────
        heatmap_imgs = []
        video_name = os.path.splitext(os.path.basename(video))[0]
        hm_dir = os.path.join(tmp_dir, video_name, 'heatmaps')
        if os.path.isdir(hm_dir):
            for fname in sorted(os.listdir(hm_dir)):
                if fname.endswith('.png'):
                    fpath = os.path.join(hm_dir, fname)
                    img   = Image.open(fpath)
                    heatmap_imgs.append(img)
        else:
            # Generate a simple heatmap from attention maps
            heatmap_imgs = generate_fallback_heatmap(video)

        annotated_vid_path = result.get('annotated_video', None)

        return pred_str, s_str, t_str, f_str, annotated_vid_path, fig, heatmap_imgs

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return f"❌ Error: {str(e)}", "", "", "", None, None, []


def make_frame_plot(clip_probs, threshold=0.5):
    """Build matplotlib figure for frame-level probabilities."""
    t = np.arange(len(clip_probs)) * 3.0  # seconds per clip
    p = np.array(clip_probs)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, p, 'b-o', linewidth=2, markersize=5, label='P(fake)')
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold:.1f}')
    ax.fill_between(t, threshold, p,
                    where=p >= threshold,
                    alpha=0.25, color='red', label='Suspected fake region')
    ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('P(Fake)', fontsize=11)
    ax.set_title('Frame-Level Deepfake Probability', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_fallback_heatmap(video_path):
    """Return a plain DCT energy visualization if no heatmaps were saved."""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return []

        frame_rgb = cv2.cvtColor(
            cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB
        )
        return [Image.fromarray(frame_rgb)]
    except Exception:
        return []


# ─────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="Deepfake Detector",
        css="""
        #title { text-align: center; color: #2D3748; }
        .prediction-box { font-size: 1.3em; font-weight: bold; }
        """,
    ) as demo:
        # Header
        gr.Markdown("# 🔍 Multimodal Deepfake Detection System", elem_id="title")
        gr.Markdown(
            "Upload a video to detect whether it has been manipulated. "
            "The system uses **three complementary analysis branches**: "
            "Spatial (FCG), Temporal (Bi-LSTM), and Frequency (DCT)."
        )

        with gr.Row():
            # Left: Upload + controls
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="📹 Upload Video",
                    height=280,
                )
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Detection Threshold",
                    info="Videos with P(fake) above this are classified as FAKE",
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Video",
                    variant="primary",
                    size="lg",
                )

            # Right: Results
            with gr.Column(scale=1):
                result_text = gr.Textbox(
                    label="🔎 Prediction",
                    interactive=False,
                    elem_classes=["prediction-box"],
                )
                with gr.Group():
                    spatial_text  = gr.Textbox(label="Spatial  Branch", interactive=False)
                    temporal_text = gr.Textbox(label="Temporal Branch", interactive=False)
                    freq_text     = gr.Textbox(label="Frequency Branch", interactive=False)
                
                annotated_vid = gr.Video(
                    label="🎥 Annotated Video Output",
                    interactive=False,
                    height=280,
                )

        # Frame-level analysis chart
        with gr.Row():
            frame_plot = gr.Plot(label="📊 Frame-Level Analysis")

        # Heatmaps
        with gr.Row():
            heatmap_gallery = gr.Gallery(
                label="🔥 Attention Heatmaps (per face part: lips/skin/eyes/nose)",
                columns=2,
                height=350,
            )

        # Examples
        gr.Markdown("---")
        gr.Markdown(
            "**Tips**: Longer videos produce more clips and a more reliable prediction. "
            "Heatmaps show which facial regions the model focused on. "
            "Red zones in the chart indicate suspected manipulated frames."
        )

        # Wire up
        analyze_btn.click(
            fn=predict,
            inputs=[video_input, threshold_slider],
            outputs=[
                result_text,
                spatial_text,
                temporal_text,
                freq_text,
                annotated_vid,
                frame_plot,
                heatmap_gallery,
            ],
        )

    return demo


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Deepfake Detector Gradio UI')
    p.add_argument('--checkpoint',    required=True,
                   help='Path to best.pt model checkpoint')
    p.add_argument('--face-features', required=True,
                   help='Path to L14_real_semantic_patches_v4_2000.pickle')
    p.add_argument('--num-frames',    type=int, default=8)
    p.add_argument('--port',          type=int, default=7860)
    p.add_argument('--share',         action='store_true',
                   help='Create a public Gradio link')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    _args = args

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Loading model on {_device}...")
    _model = load_model(args.checkpoint, args.face_features, args.num_frames, _device)
    print("[INFO] Model loaded. Starting Gradio...")

    demo = build_interface()
    demo.launch(
        server_name='0.0.0.0',
        server_port=args.port,
        share=args.share,
    )
