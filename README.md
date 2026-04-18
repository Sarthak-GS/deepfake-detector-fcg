# 🎬 Deepfake Detection System: A Multimodal Approach

A state-of-the-art video deepfake detection system that combines three complementary analysis branches for robust fake video detection:

- **Spatial (FCG)** — Facial Component Guidance attention mechanism on OpenCLIP ViT-L/14 features
- **Temporal (Bi-LSTM)** — Bidirectional LSTM analyzes temporal consistency across video frames
- **Frequency (DCT)** — Block-level Discrete Cosine Transform detects compression artifacts

The model is trained on state-of-the-art deepfake datasets (**DF40**, **FaceForensics++**, and **FaceShifter**) and achieves excellent generalization across cross-dataset evaluation.

---

## 📁 Project Structure

```
deepfake_detector_project/
├── train.py                          # Training script with detailed hyperparameters
├── test.py                           # Evaluation script for metrics
├── inference.py                      # Single-video inference with visualizations
├── app.py                            # Gradio web UI for interactive predictions
├── prepare_user_dataset.py           # Script to prepare custom unstructured datasets
├── trim_videos.py                    # Video trimming utility (optional preprocessing)
├── environment.yml                   # Conda environment specification
├── configs/
│   └── config.yaml                   # Hyperparameter configuration file
├── src/
│   ├── model/
│   │   ├── model.py                  # EnhancedDeepfakeDetector (main model)
│   │   ├── temporal_lstm.py         # Bi-LSTM temporal branch implementation
│   │   └── frequency_dct.py         # DCT frequency analysis branch
│   ├── dataset/
│   │   └── dataset.py               # Dataset loaders (FF++, CDF, simple formats)
│   └── preprocess/
│       └── preprocess.py            # Face detection, landmark extraction, cropping
├── misc/
│   ├── 20words_mean_face.npy        # Mean face template for alignment
│   └── L14_real_semantic_patches_v4_2000.pickle  # Pre-extracted ViT-L/14 features
└── checkpoints/                      # Trained model checkpoints (saved during training)
## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment from specification
conda env create -f environment.yml

# Activate the environment
conda activate deepfake_detector

# (Optional) Install additional dependencies if needed
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch==2.24.0
pip install face-alignment==1.4.1
pip install gradio==4.31.0
pip install scikit-learn==1.3.2 scipy==1.10.1
```

### 2. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 3060 (12GB VRAM) | RTX 4090 / A100 (40GB+ VRAM) |
| **CPU RAM** | 32GB | 64GB |
| **Storage** | 150GB (dataset + models) | 300GB+ |
| **CUDA** | 11.8+ | 12.1+ |

---

## 📊 Datasets

### **Supported Datasets**

| Dataset | Type | Size | Usage | Link |
|---------|------|------|-------|------|
| **DF40** | Fake faces | ~20GB (trimmed) | Training | [GitHub](https://github.com/YZY-stack/DF40) / [Videos](https://drive.google.com/drive/folders/1GB3FN4pjf9Q5hhhcBmBTdMmEmtrDe9zZ) |
| **FaceForensics++** (c23) | Multi-method manipulation | ~40GB | Training & Testing | [Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23) / [Official](https://github.com/ondyari/FaceForensics) |
| **FaceShifter** | Face-swap detection | ~5GB | Cross-validation | Part of FF++ |
| **Celeb-DF v2** | High-quality synthetic videos | ~15GB | Cross-dataset evaluation | [Download](http://www.cs.albany.edu/~lsw/celeb-df.html) |

### **FaceForensics++ (FF++) Folder Structure**

```
datasets/ffpp/
├── real/c23/videos/
│   ├── xxx.avi          # Real video 1
│   ├── yyy.avi          # Real video 2
│   └── ...
├── DF/c23/videos/       # DeepFaces manipulation
│   ├── aaa.avi
│   └── ...
├── FS/c23/videos/       # FaceShifter face-swap
│   ├── bbb.avi
│   └── ...
├── F2F/c23/videos/      # Face2Face manipulation
│   ├── ccc.avi
│   └── ...
├── NT/c23/videos/       # NeuralTextures manipulation
│   ├── ddd.avi
│   └── ...
└── csv_files/
    ├── train.json       # [[id1, id2], ...] for 720 real-fake pairs
    ├── val.json         # 140 pairs for validation
    └── test.json        # 140 pairs for testing
```

### **Expected Dataset Directory After Training**

After running preprocessing, the system creates:

```
datasets/ffpp_processed/
├── real/
│   ├── video_001/
│   │   └── frames/      # Extracted frames
│   └── video_002/
├── fake/
│   ├── video_001/
│   │   └── frames/
│   └── video_002/
└── landmarks/           # Cached facial landmarks (.json)
```

---

## 🛠️ Data Preparation Guide

### **Option 1: Structured Dataset (FF++, DF40)**

For datasets organized in the standard folder structure:

```bash
python train.py \
    --ffpp-root /path/to/ffpp \
    --cdf-root /path/to/cdf \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/run1 \
    --batch-size 4 \
    --epochs 30 \
    --num-frames 8 \
    --num-workers 4
```

The script will automatically:
1. **Detect** facial landmarks using face-alignment library
2. **Crop** and align faces using the mean face template
3. **Extract** features from pre-trained ViT-L/14 backbone
4. **Cache** processed data for faster subsequent runs

### **Option 2: Unstructured Dataset (Custom Videos)**

If you have **videos in a folder without structured labels**, use the `prepare_user_dataset.py` script:

#### **Step 1: Organize Your Videos**

Create a simple folder structure:

```
my_data/
├── fake/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
```

*Alternatively*, if all videos are in one folder, use the `prepare_user_dataset.py` script with a CSV manifest:

```
videos_folder/
├── video1.mp4
├── video2.mp4
├── video3.mp4
└── labels.csv
```

Where `labels.csv`:
```csv
filename,label
video1.mp4,fake
video2.mp4,real
video3.mp4,fake
```

#### **Step 2: Prepare the Dataset**

```bash
python prepare_user_dataset.py \
    --input-dir my_data/ \
    --output-dir processed_data/ \
    --num-workers 4
```

**Arguments:**
- `--input-dir` (required): Path to folder with `fake/` and `real/` subfolders
- `--output-dir` (default: `processed_data/`): Where to save processed videos
- `--num-workers` (default: 4): Parallel processing threads
- `--trim-videos` (optional flag): Enable video trimming (see below)
- `--trim-duration` (default: 10): Duration in seconds for each trimmed segment

After this, train using:

```bash
python train.py \
    --dataset-type simple \
    --ffpp-root processed_data/ \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --skip-prep \
    --output-dir checkpoints/custom_run \
    --batch-size 4 \
    --epochs 30 \
    --num-frames 8
```

---

## ✂️ Video Trimming (Optional)

Videos can be long and redundant. The system provides optional trimming:

### **Trim Videos Before Training**

```bash
python trim_videos.py \
    --input-dir datasets/ffpp/ \
    --output-dir datasets/ffpp_trimmed/ \
    --trim-duration 10 \
    --num-workers 4
```

**Arguments:**
- `--input-dir` (required): Source video directory
- `--output-dir` (required): Destination for trimmed videos
- `--trim-duration` (default: 10): Seconds to keep from each video
- `--num-workers` (default: 4): Parallel processing threads
- `--start-offset` (default: 0): Start trimming from this second

**Output:** Trimmed videos in the same folder structure

### **Train with Trimmed Videos**

```bash
python train.py \
    --ffpp-root datasets/ffpp_trimmed/ \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/run_trimmed \
    --batch-size 4 \
    --epochs 30
```

**Why trim?**
- ✅ Reduces storage requirements (10 sec video = ~500MB vs. full = 2GB)
- ✅ Faster data loading during training
- ✅ Maintains representative content from each video
- ✅ Still captures sufficient spatial-temporal patterns

---

## 📥 Pre-trained Checkpoint

### **Download Pre-trained Model**

We provide a pre-trained model checkpoint trained on DF40 + FaceForensics++ + FaceShifter datasets:

**📌 Checkpoint Download Link:** [https://drive.google.com/drive/folders/1rKCkIgExfYjbyKsz3LCHr4wIlpzJt7BR](https://drive.google.com/drive/folders/1rKCkIgExfYjbyKsz3LCHr4wIlpzJt7BR)

### **Setup Instructions**

#### **Step 1: Download the Checkpoint**

1. Click the [Google Drive link](https://drive.google.com/drive/your-checkpoint-link)
2. Download the file: `best.pt`

#### **Step 2: Place in Correct Directory**

Create the checkpoint directory and place the downloaded file:

```bash
# Create checkpoints folder if it doesn't exist
mkdir -p checkpoints/pretrained

# Move your downloaded best.pt file here
# Using your file manager or command line:
mv ~/Downloads/best.pt checkpoints/pretrained/
```

**Final structure:**
```
deepfake_detector_project/
├── checkpoints/
│   └── pretrained/
│       └── best.pt          ← Your downloaded checkpoint
└── ...
```

---

## 🎯 Inference: Analyze Deepfakes

### **Two Ways to Use the Model**

After downloading the pre-trained checkpoint, you have two options:

| Method | Use Case | Command |
|--------|----------|---------|
| **🖥️ Command Line (CLI)** | Single video analysis, batch processing, integration | `python inference.py` |
| **🌐 Gradio Web UI** | Interactive interface, no coding needed, easy sharing | `python app.py` |

### **Option 1: Command-Line Inference (CLI)**

#### **Basic Analysis**

```bash
python inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --save-heatmaps \
    --save-plot \
    --output-dir inference_results/
```

#### **Inference Arguments**

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--video` | *required* | Path to input video file (.mp4, .avi, .mov) | `video.mp4` |
| `--checkpoint` | *required* | Path to trained model | `checkpoints/pretrained/best.pt` |
| `--face-features` | *required* | Path to ViT-L/14 features | `misc/L14_real_semantic_patches_v4_2000.pickle` |
| `--output-dir` | `inference_results` | Where to save results | `results/` |
| `--num-frames` | 8 | Frames per clip | `12` |
| `--clip-duration` | 3.0 | Seconds per clip | `5.0` |
| `--stride-sec` | 1.0 | Sliding window stride in seconds | `2.0` |
| `--save-heatmaps` | False | Save attention heatmaps as images | `--save-heatmaps` |
| `--save-plot` | False | Save frame-level probability chart | `--save-plot` |
| `--save-video` | False | Save annotated output video with boxes | `--save-video` |

#### **Output Sample**

```
==================================================
  PREDICTION  : FAKE
  CONFIDENCE  : 87.3%
  
  Branch Scores:
    Spatial     : 0.8912
    Temporal    : 0.9103
    Frequency   : 0.8241
  
  Clips Analyzed : 4
  Average Clip Confidence : 0.8752
==================================================

Saved outputs:
  ✓ results.json         (detailed predictions)
  ✓ heatmaps/            (attention visualizations)
  ✓ frame_probs.png      (temporal confidence chart)
  ✓ output_annotated.mp4 (video with bounding boxes)
```

#### **Example 1: Detailed Analysis with Visualizations**

```bash
python inference.py \
    --video dataset/fake_video.mp4 \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --clip-duration 2.0 \
    --stride-sec 0.5 \
    --save-heatmaps \
    --save-plot \
    --save-video \
    --output-dir detailed_results/
```

#### **Example 2: Fast Inference**

```bash
python inference.py \
    --video video.mp4 \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --clip-duration 5.0 \
    --stride-sec 2.0 \
    --output-dir fast_results/
```

### **Option 2: Web UI (Gradio)**

#### **Launch Interactive UI**

```bash
python app.py \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --port 7860
```

**Then open in browser:** Go to `http://localhost:7860`

#### **App Arguments**

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--checkpoint` | *required* | Model checkpoint | `checkpoints/pretrained/best.pt` |
| `--face-features` | *required* | ViT features | `misc/L14_real_semantic_patches_v4_2000.pickle` |
| `--port` | 7860 | Server port | `--port 8080` |
| `--share` | False | Create public Gradio link | `--share` |

#### **UI Features**

- 📤 Upload video file (mp4, avi, mov)
- 🎯 Real-time prediction with confidence
- 📊 Per-branch scores (Spatial, Temporal, Frequency)
- 📈 Frame-level probability chart
- 🔍 Attention heatmaps for face parts
- 💾 Download results as JSON

---

## 🏋️ Train Your Own Model

Want to train on your own dataset? Follow these steps:

### **1. Prepare Your Dataset**

Choose one of two options:

**Option A: Structured Dataset (FF++, DF40)**
```
datasets/ffpp/
├── real/c23/videos/
├── DF/c23/videos/
├── FS/c23/videos/
├── F2F/c23/videos/
├── NT/c23/videos/
└── csv_files/
    ├── train.json
    ├── val.json
    └── test.json
```

**Option B: Unstructured Custom Videos**
```
my_data/
├── fake/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── real/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

For Option B, prepare using:
```bash
python prepare_user_dataset.py \
    --input-dir my_data/ \
    --output-dir processed_data/ \
    --num-workers 4
```

### **2. Start Training**

Using your prepared dataset:

```bash
python train.py \
    --ffpp-root /path/to/data \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/my_run \
    --batch-size 4 \
    --epochs 30 \
    --num-frames 8 \
    --num-workers 4
```

**For unstructured data:**
```bash
python train.py \
    --dataset-type simple \
    --ffpp-root processed_data/ \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --skip-prep \
    --output-dir checkpoints/my_run \
    --batch-size 4 \
    --epochs 30 \
    --num-frames 8 \
    --num-workers 4
```

### **3. Monitor Training**

- **Checkpoints saved to:** `checkpoints/my_run/`
  - `best.pt` — Best validation performance (use for inference)
  - `last.pt` — Latest epoch (use to resume interrupted training)
  
- **Resume interrupted training:**
```bash
python train.py ... --resume checkpoints/my_run/last.pt --epochs 60
```

### **4. Test Your Model**

Evaluate on test sets:

```bash
python test.py \
    --checkpoint checkpoints/my_run/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --ffpp-root /path/to/test/data \
    --output-dir results/
```

---

## 📚 Detailed Training Reference

### **Basic Training Command**

```bash
python train.py \
    --ffpp-root /data1/datasets/ffpp_trimmed/ \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/run1 \
    --batch-size 4 \
    --epochs 30 \
    --num-frames 8 \
    --num-workers 4
```

### **Detailed Argument Explanation**

#### **Required Arguments**

| Argument | Description | Example |
|----------|-------------|---------|
| `--ffpp-root` | Path to FaceForensics++ dataset with `real/`, `DF/`, `FS/`, etc. folders | `/data/ffpp` |
| `--face-features` | Path to pre-extracted ViT-L/14 semantic patches | `misc/L14_real_semantic_patches_v4_2000.pickle` |

#### **Dataset Arguments**

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--dataset-type` | `ffpp` | Dataset format: `ffpp` (structured) or `simple` (fake/real subfolders) | `--dataset-type simple` |
| `--cdf-root` | `None` | Path to Celeb-DF v2 dataset (for cross-val) | `--cdf-root /data/cdf` |
| `--skip-prep` | False | Skip automatic preprocessing (landmarks + face cropping) if already done | `--skip-prep` |
| `--limit-samples` | `None` | Limit dataset size for quick testing (e.g., 50 videos) | `--limit-samples 50` |

#### **Training Hyperparameters**

| Argument | Default | Description | Range | Example |
|----------|---------|-------------|-------|---------|
| `--batch-size` | 3 | Videos per batch (higher = more GPU memory) | 1-32 | `--batch-size 8` |
| `--epochs` | 30 | Total training epochs | 10-100 | `--epochs 50` |
| `--lr` | 1e-4 | Learning rate for AdamW optimizer | 1e-5 to 1e-3 | `--lr 0.0001` |
| `--weight-decay` | 1e-3 | L2 regularization weight | 0 to 0.1 | `--weight-decay 0.001` |
| `--num-frames` | 8 | Frames sampled per video clip | 4-16 | `--num-frames 12` |
| `--num-workers` | 4 | Parallel CPU threads for data loading | 0-12 | `--num-workers 8` |
| `--accum-steps` | 2 | Gradient accumulation steps (effective batch = batch-size × accum-steps) | 1-4 | `--accum-steps 4` |

#### **Loss & Optimization**

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--auc-weight` | 0.7 | Weight of AUC in composite score (0.7 AUC + 0.3 loss) | `--auc-weight 0.8` |
| `--min-epochs` | 5 | Warm-up: early stopping disabled for first N epochs | `--min-epochs 10` |
| `--smooth-window` | 3 | Average validation score over last N epochs for comparison | `--smooth-window 5` |
| `--patience` | 15 | Early stopping patience: stop if no improvement for N epochs | `--patience 20` |

#### **Model & Checkpoint**

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--output-dir` | `checkpoints` | Directory to save checkpoints | `--output-dir results/exp1` |
| `--resume` | `None` | Resume from checkpoint (e.g., `last.pt`) | `--resume checkpoints/run1/last.pt` |
| `--seed` | 1019 | Random seed for reproducibility | `--seed 42` |
| `--no-amp` | False | Disable mixed precision (fp16) training | `--no-amp` |

#### **Configuration File**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/config.yaml` | YAML file with all hyperparameters (CLI args override file) |

### **Advanced Training Examples**

#### **Example 1: Trimmed Data with Simple Dataset Format**

```bash
python train.py \
    --dataset-type simple \
    --ffpp-root /data1/pipesk/data_full_trimmed_cropped \
    --skip-prep \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/full_run1 \
    --batch-size 4 \
    --epochs 30 \
    --num-frames 4 \
    --num-workers 4
```

**What this does:**
- `--dataset-type simple`: Expects `/data_full_trimmed_cropped/fake/` and `/data_full_trimmed_cropped/real/` folders
- `--skip-prep`: Skips landmark detection (already processed)
- `--batch-size 4`: 4 videos per batch
- `--num-frames 4`: Sample 4 frames per video (good for already-preprocessed data)
- Saves best model to `checkpoints/full_run1/best.pt`

#### **Example 2: Full FaceForensics++ with Cross-Dataset Validation**

```bash
python train.py \
    --dataset-type ffpp \
    --ffpp-root /data/ffpp \
    --cdf-root /data/cdf \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/ffpp_full \
    --batch-size 8 \
    --lr 0.0001 \
    --epochs 50 \
    --num-frames 12 \
    --num-workers 8 \
    --accum-steps 2 \
    --weight-decay 0.001 \
    --patience 20 \
    --auc-weight 0.7
```

**Arguments explained:**
- `--batch-size 8`: Larger batch for better gradient estimates
- `--lr 0.0001`: Standard learning rate
- `--num-frames 12`: More frames = better temporal coverage
- `--accum-steps 2`: Gradient accumulation for larger effective batch
- `--patience 20`: More patience during training
- `--auc-weight 0.7`: Optimize for AUC (70%) over loss (30%)

#### **Example 3: Resume Training from Checkpoint**

```bash
python train.py \
    --ffpp-root datasets/ffpp \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/run1 \
    --batch-size 4 \
    --epochs 100 \
    --resume checkpoints/run1/last.pt
```

**Continues training:** Picks up from epoch 30 (if trained for 30 before), continues to epoch 100

#### **Example 4: Using YAML Configuration**

Create `configs/config.yaml`:
```yaml
batch_size: 8
epochs: 30
lr: 0.0001
weight_decay: 0.001
num_frames: 12
num_workers: 8
accum_steps: 2
patience: 15
auc_weight: 0.7
min_epochs: 5
smooth_window: 3
```

Then run:
```bash
python train.py \
    --config configs/config.yaml \
    --ffpp-root /data/ffpp \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/yaml_run
```

CLI arguments override config file values.

---

## 📈 Evaluation & Testing

### **Test on FF++ and Celeb-DF**

```bash
python test.py \
    --checkpoint checkpoints/full_run1/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --ffpp-root /data/ffpp \
    --cdf-root /data/cdf \
    --output-dir results/ \
    --batch-size 4 \
    --num-frames 8 \
    --num-workers 4
```

### **Test Arguments**

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--checkpoint` | *required* | Path to trained model (`best.pt`) | `checkpoints/run1/best.pt` |
| `--face-features` | *required* | Path to ViT-L/14 features | `misc/L14_real_semantic_patches_v4_2000.pickle` |
| `--ffpp-root` | `None` | Path to FF++ test set (optional) | `/data/ffpp` |
| `--cdf-root` | `None` | Path to Celeb-DF (optional, cross-dataset eval) | `/data/cdf` |
| `--output-dir` | `results` | Where to save JSON results | `results/` |
| `--batch-size` | 4 | Batch size for inference | `8` |
| `--num-frames` | 8 | Frames per clip | `12` |
| `--num-workers` | 4 | Data loading threads | `8` |

### **Expected Performance**

| Metric | FF++ (In-Domain) | Celeb-DF v2 (Cross-Dataset) |
|--------|------------------|---------------------------|
| **AUC** | > 0.98 | > 0.85 |
| **Accuracy** | > 0.95 | > 0.80 |
| **F1-Score** | > 0.95 | > 0.75 |

---

## � Complete Command Reference

### **Setup Instructions**

#### **Step 1: Download the Checkpoint**

1. Click the [Google Drive link](https://drive.google.com/drive/your-checkpoint-link)
2. Download the file: `best.pt`

#### **Step 2: Place in Correct Directory**

Create the checkpoint directory and place the downloaded file:

```bash
# Create checkpoints folder if it doesn't exist
mkdir -p checkpoints/pretrained

# Move your downloaded best.pt file here
# Using your file manager or command line:
mv ~/Downloads/best.pt checkpoints/pretrained/
```

**Final structure:**
```
deepfake_detector_project/
├── checkpoints/
│   └── pretrained/
│       └── best.pt          ← Your downloaded checkpoint
└── ...
```

#### **Step 3: Use for Inference**

Once downloaded, use the checkpoint for inference (no training needed):

```bash
python inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --save-heatmaps \
    --save-plot \
    --output-dir inference_results/
```

---

## 🎯 Inference: Analyze Deepfakes

### **Two Ways to Use the Model**

After downloading the pre-trained checkpoint, you have two options:

| Method | Use Case | Command |
|--------|----------|---------|
| **🖥️ Command Line (CLI)** | Single video analysis, batch processing, integration | `python inference.py` |
| **🌐 Gradio Web UI** | Interactive interface, no coding needed, easy sharing | `python app.py` |

---

### **Option 1: Command-Line Inference (CLI)**

For programmatic video analysis and batch processing:

#### **Basic Analysis**

```bash
python inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --save-heatmaps \
    --save-plot \
    --output-dir inference_results/
```

### **Inference Arguments**

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--video` | *required* | Path to input video file (.mp4, .avi, .mov) | `video.mp4` |
| `--checkpoint` | *required* | Path to trained model | `checkpoints/best.pt` |
| `--face-features` | *required* | Path to ViT-L/14 features | `misc/L14_real_semantic_patches_v4_2000.pickle` |
| `--output-dir` | `inference_results` | Where to save results | `results/` |
| `--num-frames` | 8 | Frames per clip | `12` |
| `--clip-duration` | 3.0 | Seconds per clip | `5.0` |
| `--stride-sec` | 1.0 | Sliding window stride in seconds | `2.0` |
| `--save-heatmaps` | False | Save attention heatmaps as images | `--save-heatmaps` |
| `--save-plot` | False | Save frame-level probability chart | `--save-plot` |
| `--save-video` | False | Save annotated output video with boxes | `--save-video` |

### **Output Sample**

```
==================================================
  PREDICTION  : FAKE
  CONFIDENCE  : 87.3%
  
  Branch Scores:
    Spatial     : 0.8912
    Temporal    : 0.9103
    Frequency   : 0.8241
  
  Clips Analyzed : 4
  Average Clip Confidence : 0.8752
==================================================

Saved outputs:
  ✓ results.json         (detailed predictions)
  ✓ heatmaps/            (attention visualizations)
  ✓ frame_probs.png      (temporal confidence chart)
  ✓ output_annotated.mp4 (video with bounding boxes)
```

### **Advanced Inference Examples**

#### **Example 1: Detailed Analysis with Visualizations**

```bash
python inference.py \
    --video dataset/fake_video.mp4 \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --clip-duration 2.0 \
    --stride-sec 0.5 \
    --save-heatmaps \
    --save-plot \
    --save-video \
    --output-dir detailed_results/
```

**Parameters:**
- `--clip-duration 2.0`: Shorter 2-second clips = more clips analyzed
- `--stride-sec 0.5`: Overlapping windows (dense analysis)
- `--save-heatmaps`: Generate attention maps for each face part
- `--save-plot`: Generate frame-confidence timeline
- `--save-video`: Annotate original video with predictions

#### **Example 2: Fast Inference**

```bash
python inference.py \
    --video video.mp4 \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --clip-duration 5.0 \
    --stride-sec 2.0 \
    --output-dir fast_results/
```

**Parameters:** Longer clips and stride = fewer predictions = faster

---

---

### **Option 2: Web UI (Gradio)**

For interactive, user-friendly interface with no coding:

#### **Launch Interactive UI**

```bash
python app.py \
    --checkpoint checkpoints/pretrained/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --port 7860
```

**Then open in browser:** Go to `http://localhost:7860`

#### **App Arguments**

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--checkpoint` | *required* | Model checkpoint | `checkpoints/pretrained/best.pt` |
| `--face-features` | *required* | ViT features | `misc/L14_real_semantic_patches_v4_2000.pickle` |
| `--port` | 7860 | Server port | `--port 8080` |
| `--share` | False | Create public Gradio link | `--share` |

#### **UI Features**

- 📤 Upload video file (mp4, avi, mov)
- 🎯 Real-time prediction with confidence
- 📊 Per-branch scores (Spatial, Temporal, Frequency)
- 📈 Frame-level probability chart
- 🔍 Attention heatmaps for face parts
- 💾 Download results as JSON

---

## 🏋️ Train Your Own Model

Want to train on your own dataset? Follow these steps:

### **1. Prepare Your Dataset**

Choose one of two options:

**Option A: Structured Dataset (FF++, DF40)**
```
datasets/ffpp/
├── real/c23/videos/
├── DF/c23/videos/
├── FS/c23/videos/
├── F2F/c23/videos/
├── NT/c23/videos/
└── csv_files/
    ├── train.json
    ├── val.json
    └── test.json
```

**Option B: Unstructured Custom Videos**
```
my_data/
├── fake/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── real/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

For Option B, prepare using:
```bash
python prepare_user_dataset.py \
    --input-dir my_data/ \
    --output-dir processed_data/ \
    --num-workers 4
```

### **2. Start Training**

Using your prepared dataset:

```bash
python train.py \
    --ffpp-root /path/to/data \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/my_run \
    --batch-size 4 \
    --epochs 30 \
    --num-frames 8 \
    --num-workers 4
```

**For unstructured data:**
```bash
python train.py \
    --dataset-type simple \
    --ffpp-root processed_data/ \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --skip-prep \
    --output-dir checkpoints/my_run \
    --batch-size 4 \
    --epochs 30 \
    --num-frames 8 \
    --num-workers 4
```

### **3. Monitor Training**

- **Checkpoints saved to:** `checkpoints/my_run/`
  - `best.pt` — Best validation performance (use for inference)
  - `last.pt` — Latest epoch (use to resume interrupted training)
  
- **Resume interrupted training:**
```bash
python train.py ... --resume checkpoints/my_run/last.pt --epochs 60
```

### **4. Test Your Model**

Evaluate on test sets:

```bash
python test.py \
    --checkpoint checkpoints/my_run/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --ffpp-root /path/to/test/data \
    --output-dir results/
```

### **All Training Arguments Reference**

See the [Complete Training Arguments](#full-training-arguments-reference) section below for detailed explanation of every parameter.

---

## 🔧 Complete Command Reference

### **Setup**
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate deepfake_detector
```

### **Data Preparation**
```bash
# Trim videos (optional)
python trim_videos.py --input-dir datasets/ffpp --output-dir datasets/ffpp_trimmed --trim-duration 10

# Prepare unstructured dataset
python prepare_user_dataset.py --input-dir my_data/ --output-dir processed_data/
```

### **Training**
```bash
# Full training
python train.py --ffpp-root datasets/ffpp --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/run1 --batch-size 4 --epochs 30 --num-frames 8

# With config file
python train.py --config configs/config.yaml --ffpp-root datasets/ffpp \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle

# Resume training
python train.py --resume checkpoints/run1/last.pt --ffpp-root datasets/ffpp \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle --epochs 50

# Quick test (debugging)
python train.py --ffpp-root datasets/ffpp --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --limit-samples 20 --batch-size 2 --epochs 5 --num-workers 0
```

### **Evaluation**
```bash
# Test on FF++ and Celeb-DF
python test.py --checkpoint checkpoints/run1/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --ffpp-root /data/ffpp --cdf-root /data/cdf
```

### **Inference**
```bash
# Single video analysis
python inference.py --video test.mp4 --checkpoint checkpoints/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --save-heatmaps --save-plot

# Batch inference
for video in videos/*.mp4; do
    python inference.py --video "$video" --checkpoint checkpoints/best.pt \
        --face-features misc/L14_real_semantic_patches_v4_2000.pickle
done
```

### **Web UI**
```bash
# Launch Gradio interface
python app.py --checkpoint checkpoints/best.pt \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle --port 7860
```

---

## 📝 Model Architecture

### **Three-Branch Ensemble Design**

The model combines three complementary detection branches that operate on different signal types:

```
Input Video (fps frames)
    ↓
├─→ [SPATIAL BRANCH]  ─────→ Facial Part Attention (FCG)
├─→ [TEMPORAL BRANCH] ─────→ Bi-LSTM Sequence Analysis
└─→ [FREQUENCY BRANCH] ────→ DCT Spectral Analysis
    ↓
Fused Predictions
    ↓
Output: P(Fake), Confidence Score, Per-branch scores
```

#### **1. Spatial Branch (Facial Component Guidance)**

- **Backbone**: OpenCLIP ViT-L/14 (frozen)
- **Input**: Face patches (cropped from detected faces)
- **Processing**:
  - Extract CLS token embeddings from ViT-L/14
  - Apply Synoptic Decoder with attention masks for 4 face parts:
    - 👄 **Lips** — Most manipulated region
    - 👁️ **Eyes** — Expression artifacts
    - 👃 **Nose** — Symmetry inconsistencies
    - 🎭 **Skin** — Texture artifacts
  - Cross-attention pooling over semantic patches
- **Output**: Per-face authenticity score + attention weights

#### **2. Temporal Branch (Bi-Directional LSTM)**

- **Input**: CLS token sequences (8-16 frames per clip)
- **Architecture**:
  - Bidirectional LSTM (256 hidden units each direction)
  - Output dimension: 512 (256 forward + 256 backward)
- **Purpose**:
  - Capture temporal inconsistencies in fake videos
  - Learn state transitions across frames
  - Detect sudden changes in facial features
- **Output**: Temporal authenticity score

#### **3. Frequency Branch (Discrete Cosine Transform)**

- **Input**: Block-DCT coefficients (spatial frequency domain)
- **Processing**:
  - Divide face into 8×8 blocks
  - Compute DCT for each block
  - Analyze frequency patterns for compression artifacts
  - Deepfakes show distinct frequency signatures due to GAN generation
- **Output**: Frequency authenticity score

### **Fusion Strategy**

```python
final_score = w_spatial * spatial_logit + \
              w_temporal * temporal_logit + \
              w_frequency * frequency_logit
```

**Trainable fusion weights** learned during training to optimally combine branch predictions.

---

## 📊 Training Statistics

### **Training Dynamics**

| Metric | Value |
|--------|-------|
| **Trainable Parameters** | ~14.6M |
| **Frozen Parameters** | ~304M (ViT-L backbone) |
| **Training Time (FF++)** | 6-8 hours (48GB GPU) |
| **Memory per Batch** | ~8GB for batch_size=4 |

### **Convergence**

- **Early Stopping**: Enabled after `--min-epochs` warmup
- **Patience**: 15 epochs (default)
- **Best Model**: Selected by composite score (AUC vs. normalized loss)
- **Checkpoints Saved**:
  - `best.pt` — Best validation performance
  - `last.pt` — Latest epoch (for resuming)

### **Loss Function**

```
Total Loss = Focal Classification Loss + FCG Alignment Loss

Classification Loss = Focal((1 - p_t)^γ * log(p_t), γ=4.0)
FCG Alignment Loss = Cosine Similarity Loss (soft alignment, temp=30)

Weighted: 10.0 × cls_loss + 1.5 × ffg_loss
```

---

## 🧪 Reproducibility

### **Random Seed**

```bash
# All results use seed 1019 for reproducibility
python train.py ... --seed 1019
```

- Sets `random.seed()`, `np.random.seed()`, `torch.manual_seed()`
- Enables deterministic CUDA operations
- Ensures same split order and data loading sequence

### **Config for Reproducible Results**

Save this to `configs/reproducible.yaml`:

```yaml
seed: 1019
batch_size: 4
epochs: 30
lr: 0.0001
weight_decay: 0.001
num_frames: 8
num_workers: 4
accum_steps: 2
patience: 15
auc_weight: 0.7
min_epochs: 5
smooth_window: 3
```

Then run:
```bash
python train.py --config configs/reproducible.yaml \
    --ffpp-root datasets/ffpp \
    --face-features misc/L14_real_semantic_patches_v4_2000.pickle \
    --output-dir checkpoints/reproducible_run
```

## 🐛 Troubleshooting

### Q: `CUDA out of memory` error during inference/training
**A:** This happens when batch size is too large for your GPU. Solutions:
```bash
# Reduce batch size in config.yaml
batch_size: 8  # Instead of 16 or 32

# Or enable mixed precision (fp16) in training
# Already configured in train.py for memory efficiency
```

### Q: `No face detected` error during preprocessing
**A:** This occurs when `face-alignment` fails to detect faces:
```bash
# Try lower confidence threshold in preprocess.py
# Or manually check if video contains clear frontal faces
# Some synthetic videos may have detection artifacts

# Verify video format
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate your_video.mp4
```

### Q: `RuntimeError: NCCL operation failed` during distributed training
**A:** NCCL communication issue on multi-GPU setup:
```bash
# Use single GPU instead
export CUDA_VISIBLE_DEVICES=0
python train.py

# Or check GPU connectivity
nvidia-smi -q -d PCIE
```

### Q: Gradio app fails to load checkpoint
**A:** Checkpoint path or GPU memory issue. Debug with:
```bash
# Test checkpoint loading directly
python -c "import torch; ckpt = torch.load('checkpoints/your_model.pth'); print('✅ Checkpoint OK')"

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Q: Very low accuracy on custom test videos
**A:** Domain shift issue. The model is trained on FF++ and may not generalize well:
- Ensure videos are 720p or higher resolution (like FF++)
- Check lighting conditions (model expects face-frontal videos)
- Verify video codec matches (H.264 recommended)
- Fine-tune on your domain: `python train.py --use_pretrained --num_epochs 5 --lr 1e-5`

### Q: Out of disk space errors when preparing datasets
**A:** FF++ is large (~370GB for c23). Solutions:
```bash
# Check disk space
df -h

# Use subset instead
python prepare_user_dataset.py --dataset_path /path/to/subset --output_path ./data

# Or stream from external drive
ln -s /mnt/external/FaceForensics++ ./data/FF++
```

### Q: Training hangs with multiprocessing errors or slow data loading
**A:** DataLoader `num_workers` issue. Too many workers can cause deadlocks:
```bash
# Reduce num_workers in config.yaml or command line
python train.py --num_workers 2  # Default is 4, try 0, 1, or 2

# Or disable multiprocessing entirely
python train.py --num_workers 0
```
**Tip:** If you notice slow data loading, try increasing `num_workers` (e.g., 4-8), but reduce if you see "worker dead" errors.

### Q: Training is too slow or want to skip preprocessing
**A:** Use preprocessed data cache to skip face detection/alignment steps:
```bash
# First run with preprocessing (one-time cost)
python train.py --preprocess_cache ./data/cache

# Subsequent runs will use cached faces instead of reprocessing
# Or disable preprocessing entirely if data is pre-aligned
python train.py --skip_preprocess
```
**Tip:** Check for `--skip_preprocess` and `--use_cache` flags in `train.py --help` for your version.

---

**Last Updated**: April 19, 2026  
**Version**: 2.0  
**Status**: Stable ✅

### **Authors**

- **Sarthak Gopal Sharma** — Lead Developer, Model Architecture & Training
- **Abhigyan Sharma** — Co-developer, Dataset Preparation & Evaluation

---

## 🔗 References & Acknowledgments

### **Inspiration & Core Techniques**

- **[DFD-FCG](https://github.com/aiiu-lab/DFD-FCG)** — Spatial branch architecture based on Facial Component Guidance (CVPR'25)
- **[OpenCLIP](https://github.com/mlfoundations/open_clip)** — Pre-trained vision-language backbone (ViT-L/14)
- **[Face Alignment Network](https://github.com/1adrianb/face-alignment)** — Facial landmark detection for preprocessing

### **Datasets Used for Training**

- **[DF40](https://github.com/YZY-stack/DF40)** — Deepfake face dataset with fake faces
- **[FaceForensics++](https://www.kaggle.com/datasets/xdxd003/ff-c23)** — Large-scale manipulation benchmark (c23 codec)
- **[FaceShifter](https://github.com/ondyari/FaceForensics)** — Face-swap detection (included in FF++)

### **Evaluation Datasets**

- **[Celeb-DF v2](http://www.cs.albany.edu/~lsw/celeb-df.html)** — High-quality synthetic videos for cross-dataset generalization

### **Key Papers**

- DFD-FCG (CVPR'25) — Synoptic decoder with facial component guidance
- Temporal consistency analysis through Bi-LSTM
- Frequency domain analysis via DCT for GAN artifact detection

---
