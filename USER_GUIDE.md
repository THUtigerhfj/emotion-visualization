# Emotion GAR - User Guide

Detailed guide for emotion recognition with CLS attention visualization and test-time register tokens.

## Table of Contents

1. [Register Neuron Discovery Pipeline](#register-neuron-discovery-pipeline)
2. [Running Inference](#running-inference)
3. [Parameter Reference](#parameter-reference)
4. [Output Files](#output-files)
5. [Troubleshooting](#troubleshooting)

---

## Register Neuron Discovery Pipeline

The discovery pipeline identifies which neurons to intervene on for Mode 3 (Registers). This is a **one-time process** - once you have `register_neurons.json`, you can use it for all future inference runs.

### Prerequisites

You need the FER2013 dataset (or any similar emotion dataset with class subdirectories):

```
data/fer2013/
├── train/
│   ├── 0/  # Angry
│   ├── 1/  # Disgust
│   ├── 2/  # Fear
│   ├── 3/  # Happy
│   ├── 4/  # Sad
│   ├── 5/  # Surprise
│   └── 6/  # Neutral
└── test/
```

### How It Works

Following the paper's approach, we discover "register neurons" at the **MLP output** (after `fc2`):

- **Discovery dimension**: 768 (hidden_size) - the MLP output space
- **Intervention point**: Same MLP output (after `fc2`)
- **Selection criterion**: Sparse, high-activating neurons that cause outlier patch norms

### Quick Start (Interactive Mode)

```bash
python src/run_complete_discovery.py \
    --image_folder data/fer2013/train \
    --num_images 1000 \
    --seed 42
```

This runs in **interactive mode**:
1. **Phase 1**: Tracks norms across all layers → generates `norm_progression.png`
2. **Prompts you** to select the explosion layer
3. **Phases 2-3**: Completes discovery and generates `register_neurons.json`

### Non-Interactive Mode

For automation or reproducibility:

```bash
python src/run_complete_discovery.py \
    --image_folder data/fer2013/train \
    --num_images 1000 \
    --seed 42 \
    --phase2_layer 10 \
    --explosion_layer 10 \
    --threshold 165 \
    --top_k_overall 10
```

### Phase-by-Phase Explanation

#### Phase 1: Norm Tracking (Find Explosion Layer)

Tracks patch norms across all ViT layers to identify where norms first spike significantly.

**Output**: `outputs/discovery/phase1_norm_tracking/norm_progression.png`

Look for the large jump in the plot. This becomes your `--explosion_layer` for Phase 2.

#### Phase 2: Extreme Patches (Find Outlier Positions)

At the `--phase2_layer` layer, identifies which patch positions have abnormally high norms.

**Outputs**:
- `patch_norm_histogram.png` - Distribution of patch norms with threshold marked. We can use it to determine the `--threshold` for outlier detection.
- `spatial_outlier_heatmap.png` - 2D heatmap showing outlier positions
- `outlier_positions.json` - List of outlier patch indices

#### Phase 3: Backward Search (Find Register Neurons)

Searches backward from the explosion layer to find neurons at the MLP output (after `fc2`, in 768-dim hidden_size space) that:
1. Have high activations at outlier patch positions
2. Are sparse (fire rarely but intensely)

**Output**: `register_neurons.json` - Map of layer indices to neuron indices. Use `--top_k_overall` to control how many neurons are selected across all layers.

Example:
```json
{
  "4": [382, 759, 187],
  "5": [187, 382, 759, 183],
  "6": [187, 382, 759]
}
```

### Parameter Reference (Discovery)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image_folder` | Required | Folder with class subdirectories (0/, 1/, ..., 6/) |
| `--num_images` | 1000 | **Total** images to sample (balanced across classes) |
| `--seed` | 42 | Random seed for reproducibility |
| `--explosion_layer` | None (prompt) | Layer where norms first spike |
| `--threshold` | None (auto) | Norm threshold for outlier detection (auto = mean + 3×std) |
| `--top_k_overall` | 10 | Number of neurons to select across all layers |
| `--output` | `models/register_neurons.json` | Output path for register neurons |

---

## Running Inference

### Understanding the Three Modes

| Mode | Description | Command Flag | Mask Enhancement | Register Tokens |
|------|-------------|--------------|-----------------|-----------------|
| **Mode 1: Vanilla** | Raw CLS attention from last layer | (default) | ❌ No | ❌ No |
| **Mode 2: Enhanced** | CLS attention + mask enhancement | `--enhance_mask` | ✅ Yes | ❌ No |
| **Mode 3: Registers** | CLS attention + test-time registers | `--use_registers` | ❌ No | ✅ Yes |

### Frontend Usage

```bash
cd frontend
python backend.py
```

The frontend supports all three modes through the UI:
- **Vanilla**: Raw CLS attention (baseline)
- **Enhanced**: Applies mask enhancement for cleaner visualization
- **Registers**: Uses test-time register intervention

### Batch Inference

```bash
# Mode 1: Vanilla (baseline)
python src/observe_batch_inference_attention.py \
    --input_dir data/observe \
    --output_dir outputs/mode1_vanilla

# Mode 2: Enhanced (mask enhancement)
python src/observe_batch_inference_attention.py \
    --input_dir data/observe \
    --output_dir outputs/mode2_enhanced \
    --enhance_mask

# Mode 3: Registers (requires register_neurons.json)
python src/observe_batch_inference_attention.py \
    --input_dir data/observe \
    --output_dir outputs/mode3_registers \
    --use_registers \
    --register_neurons models/register_neurons.json
```

### Parameter Reference (Inference)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | `data/observe` | Input image directory |
| `--output_dir` | `outputs/observe_inference_attention` | Output directory |
| `--model_dir` | `models/emotion_vit` | Local model directory |
| `--head_fusion` | `mean` | How to fuse attention heads (mean/max/min) |
| `--enhance_mask` | `False` | Apply mask enhancement (Mode 2) |
| `--use_registers` | `False` | Use register tokens (Mode 3) |
| `--register_neurons` | `models/register_neurons.json` | Path to register neurons |

---

## Output Files

### Discovery Outputs

```
outputs/discovery/
├── phase1_norm_tracking/
│   ├── norm_progression.png          # For layer selection
│   └── norm_statistics.json
├── phase2_extreme_patches/
│   ├── patch_norm_histogram.png
│   ├── spatial_outlier_heatmap.png
│   └── outlier_positions.json
└── phase3_register_neurons/
    ├── register_neurons.json         # Final output (use this for inference!)
    ├── neuron_statistics.json
    └── neuron_activation_histogram.png
```

### Inference Outputs

Each image generates a summary visualization showing:
- Original image (224×224)
- CLS attention heatmap overlay
- Predicted emotion with confidence
- True label (if organized in class folders)

Output naming: `{original_name}_summary.png`

---

## Troubleshooting

### Discovery Issues

| Issue | Solution |
|-------|----------|
| "No class subdirectories found" | Ensure images are in `data/fer2013/train/0/, 1/, ..., 6/` format |
| "Explosion layer not specified" | Either add `--explosion_layer N` or run in interactive mode |
| Out of memory | Reduce `--num_images` (e.g., 1000 → 500) |
| norm_progression.png shows no clear jump | Try increasing `--num_images` or check if your model differs from ViT-B/16 |

### Inference Issues

| Issue | Solution |
|-------|----------|
| "Model not found" | Run `python src/download_model.py` |
| "Register neurons file not found" | Run discovery first, or check path with `--register_neurons` |
| "Invalid num_patches" error | Model architecture mismatch - ensure you're using the downloaded model |
| Visualizations look identical | Ensure discovery and inference use the same model checkpoint |

### Frontend Issues

| Issue | Solution |
|-------|----------|
| "RetinaFace weights not found" | Run `python src/download_model.py` to fetch weights |
| Port 7860 already in use | Change port in `frontend/backend.py` line 273 |
| "No face detected" | Ensure image contains a clear, visible face |
| Cloudflare tunnel connection failed | Check if cloudflared is installed: `which cloudflared` |

### GPU/CUDA Issues

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Run on CPU: `CUDA_VISIBLE_DEVICES="" python src/...` |
| Model runs on CPU but GPU available | Check `torch.cuda.is_available()` returns True |
| Slow inference | Ensure GPU is being used: check `nvidia-smi` during inference |

---

## Citation

Based on:

> **Vision Transformers Don't Need Trained Registers**  
> Nick Jiang, Amil Dravid, Alexei Efros, Yossi Gandelsman  
> NeurIPS 2025 (Spotlight)  
> [arXiv:2506.08010](https://arxiv.org/abs/2506.08010)

## Repository Links

- [Original Test-Time Registers](https://github.com/nickjiang2378/test-time-registers)
- [FER2013 ViT Model](https://huggingface.co/dima806/facial_emotions_image_detection)
- [RetinaFace](https://github.com/serengil/retinaface)
