# Emotion Visualization - Emotion Recognition with Attention Visualization

Facial emotion recognition with CLS attention visualization. We proposed our own method to enhance visualization quality. We also adopted the test-time register token method, which is proposed by **"Vision Transformers Don't Need Trained Registers"** (Jiang et al., NeurIPS 2025 Spotlight).

## Features

- **7-Emotion Classification**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **CLS Attention Visualization**: See which facial regions the model focuses on for predictions
- **Test-Time Registers**: Reduce attention artifacts using training-free register tokens
- **Face Detection**: Automatic face detection and cropping with RetinaFace
- **Three Visualization Modes**: Vanilla, Enhanced, and Registers
- **Web Interface**: Modern frontend for interactive single-image analysis
- **Batch Processing**: Process entire directories of images
- **Cloudflare Tunnel Support**: Easily share your deployment publicly

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models (ViT classifier + RetinaFace)
python src/download_model.py

# 3. Run the frontend
cd frontend
python backend.py
```

Then open `http://localhost:7860` in your browser.

## Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Step 1: Create Environment

```bash
conda create -n emotion python=3.10
conda activate emotion
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` & `torchvision` - PyTorch deep learning framework
- `transformers` - HuggingFace model library
- `opencv-python` - Image processing
- `pillow` - Image I/O
- `numpy` - Numerical operations
- `gradio` - UI framework (optional, for legacy gradio app)
- `retina-face` - Face detection
- `tf-keras` - Required dependency for RetinaFace

### Step 3: Download Models

```bash
python src/download_model.py
```

This downloads:
- **ViT Emotion Model** → `models/emotion_vit/` - Fine-tuned ViT for facial emotion recognition
- **RetinaFace Weights** → `models/retinaface/retinaface.h5` - Face detection model

### Step 4: (Optional) Download FER2013 Dataset

Only needed if you want to run the register neuron discovery pipeline:

```bash
# Option A: From Kaggle (recommended)
# https://www.kaggle.com/datasets/msambare/fer2013
# Extract to data/fer2013/

# Expected structure:
# data/fer2013/
# ├── train/
# │   ├── 0/  # Angry
# │   ├── 1/  # Disgust
# │   ├── 2/  # Fear
# │   ├── 3/  # Happy
# │   ├── 4/  # Sad
# │   ├── 5/  # Surprise
# │   └── 6/  # Neutral
# └── test/
```

### Step 5: (Optional) Install Cloudflared

For sharing your deployment publicly via Cloudflare Tunnel:

```bash
# Linux (AMD64)
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Or use the included .deb file
sudo dpkg -i cloudflared-linux-amd64.deb
```

## Three Visualization Modes

| Mode | Description | Method | Use Case |
|------|-------------|--------|----------|
| **Mode 1: Vanilla** | Raw CLS attention from last layer | No post-processing | Baseline visualization |
| **Mode 2: Enhanced** | CLS attention + mask enhancement | Applies `enhance_rollout_mask` | Cleaner attention maps |
| **Mode 3: Registers** | CLS attention + test-time registers | Adds register token intervention | Reduces outlier artifacts |

**What are Register Tokens?**

Following Jiang et al. (2025), we add 1 zero-initialized register token to absorb outlier activations:
- **Discovery**: Find "register neurons" at MLP output (after fc2, in 768-dim hidden_size space)
- **Intervention**: For each register neuron, find sign_max (largest absolute value) and shift it from patch tokens to the register token
- **Result**: Cleaner attention patterns without retraining

## Usage

### Frontend (Recommended)

The modern web interface supports all three modes and real-time face detection:

```bash
cd frontend
python backend.py
```

The server runs on `http://0.0.0.0:7860`. Features:
- Drag-and-drop image upload
- Mode selection (Vanilla/Enhanced/Registers)
- Face detection with bounding boxes
- Per-face attention visualization
- Emotion distribution summary

### Share with Cloudflare Tunnel

To make your frontend publicly accessible:

```bash
# Terminal 1: Start the backend
cd frontend
python backend.py

# Terminal 2: Start Cloudflare tunnel
cloudflared tunnel --url http://localhost:7860 --protocol http2
```

Cloudflare will generate a public URL (e.g., `https://xxx-xxx-xxx.trycloudflare.com`) that you can share with others.

### Batch Inference

Process entire directories of images:

```bash
# Mode 1: Vanilla (baseline)
python src/observe_batch_inference_attention.py \
    --input_dir data/observe \
    --output_dir outputs/mode1_vanilla

# Mode 2: Enhanced (with mask enhancement)
python src/observe_batch_inference_attention.py \
    --input_dir data/observe \
    --output_dir outputs/mode2_enhanced \
    --enhance_mask

# Mode 3: Registers (with test-time registers)
python src/observe_batch_inference_attention.py \
    --input_dir data/observe \
    --output_dir outputs/mode3_registers \
    --use_registers \
    --register_neurons models/register_neurons.json
```

**Note**: Mode 3 requires `register_neurons.json`. See [USER_GUIDE.md](USER_GUIDE.md) for the register discovery pipeline.

### Register Neuron Discovery

The discovery pipeline finds which neurons to intervene on for Mode 3. See [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions.

```bash
python src/run_complete_discovery.py \
    --image_folder data/fer2013/train \
    --num_images 1000 \
    --seed 42 \
    --explosion_layer 6
```

This generates `models/register_neurons.json` for use with Mode 3.

## Project Structure

```
emotion-gar/
├── data/                          # Input data
│   ├── observe/                   # Test images for batch inference
│   └── fer2013/                   # Training data (optional, for discovery)
├── models/                         # Downloaded models
│   ├── emotion_vit/               # ViT classification model
│   ├── register_neurons.json      # Discovered register neurons
│   └── retinaface/               # Face detector weights
├── outputs/                       # Generated visualizations
│   ├── mode1_vanilla/
│   ├── mode2_enhanced/
│   └── mode3_registers/
├── frontend/                      # Web interface
│   ├── backend.py                # Flask server
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── templates/index.html
├── src/                           # Source code
│   ├── download_model.py          # Model downloader
│   ├── inference.py               # Utility functions
│   ├── observe_batch_inference_attention.py  # Batch inference
│   ├── run_complete_discovery.py             # Register discovery
│   └── vit_with_registers.py                 # Register token wrapper
├── requirements.txt
├── README.md                      # This file
└── USER_GUIDE.md                  # Detailed guide
```

## How It Works

### CLS Attention Visualization

Instead of complex attention rollout, we extract the CLS token's attention weights from the last transformer layer. This shows which image patches the model considered important for its prediction.

### Test-Time Registers

ViT models produce "outlier" patch tokens with abnormally high norms, causing attention spikes. Our solution:

1. **Discovery**: Find register neurons at MLP output (after fc2, in 768-dim hidden_size space)
2. **Intervention**: For each register neuron, find the `sign_max` (value with largest absolute value) and shift it from patch tokens to the register token
3. **Result**: Patch tokens no longer have extreme outliers → attention is smoother

### Face Detection

We use RetinaFace for:
- Automatic face detection
- Bounding box visualization
- Face cropping before emotion classification

## Citation

Based on:

> **Vision Transformers Don't Need Trained Registers**  
> Nick Jiang, Amil Dravid, Alexei Efros, Yossi Gandelsman  
> NeurIPS 2025 (Spotlight)  
> [arXiv:2506.08010](https://arxiv.org/abs/2506.08010)

## Credits

- [FER2013 ViT Model](https://huggingface.co/dima806/facial_emotions_image_detection) - By `dima806`
- [Test-Time Registers](https://github.com/nickjiang2378/test-time-registers) - By Nick Jiang et al.
- [RetinaFace](https://github.com/serengil/retinaface) - Face detection by Serengil

## License

This project is for research and educational purposes.
