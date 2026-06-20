"""
CLS Attention Batch Inference

This script uses CLS token attention weights for visualization,
exactly following the paper's approach (no attention rollout).

Three modes:
1. Mode 1 (baseline): Raw CLS attention from last layer
2. Mode 2 (enhanced): CLS attention + enhance_rollout_mask post-processing
3. Mode 3 (with registers): CLS attention with test-time register tokens
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

from inference import (
    enhance_rollout_mask,
    show_mask_on_image,
    save_image_unicode_safe,
)
from vit_with_registers import ViTWithTestTimeRegisters, load_model_with_registers


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES


def iter_images(root_dir: Path):
    for path in root_dir.rglob("*"):
        if is_image_file(path):
            yield path


def build_summary_image(original_rgb, rollout_rgb, predicted_label, confidence, true_label):
    """Build one image containing original, rollout, and class info."""
    h, w = original_rgb.shape[:2]

    info_panel = np.full((h, w, 3), 245, dtype=np.uint8)
    title = "Classification"
    pred_text = f"Predicted: {predicted_label}"
    conf_text = f"Confidence: {confidence:.2%}"
    true_text = f"Folder label: {true_label}" if true_label else "Folder label: N/A"

    cv2.putText(info_panel, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(info_panel, pred_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(info_panel, conf_text, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(info_panel, true_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)

    original_labeled = original_rgb.copy()
    rollout_labeled = rollout_rgb.copy()
    cv2.putText(original_labeled, "Original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(rollout_labeled, "CLS Attention", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return np.concatenate([original_labeled, rollout_labeled, info_panel], axis=1)


def simple_cls_attention(
    attentions,
    head_fusion="mean",
    num_registers=0,
):
    """
    Extract CLS token attention weights directly (no rollout).

    This follows the paper's approach: use only the CLS token's attention
    to image patches from the last layer.

    Args:
        attentions: List of attention tensors from each layer
            Each tensor shape: [batch_size, num_heads, seq_len, seq_len]
        head_fusion: How to fuse attention heads ("mean", "max", or "min")
        num_registers: Number of register tokens at the end of sequence

    Returns:
        mask: CLS attention mask as numpy array [H, W]
    """
    if not attentions:
        raise RuntimeError("No attention maps captured.")

    if head_fusion not in {"mean", "max", "min"}:
        raise ValueError("head_fusion must be one of: mean, max, min.")

    # Use attention from the LAST layer only (not rollout)
    attention = attentions[-1]

    # Fuse attention heads
    if head_fusion == "mean":
        attention_heads_fused = attention.mean(dim=1)
    elif head_fusion == "max":
        attention_heads_fused = attention.max(dim=1).values
    else:  # min
        attention_heads_fused = attention.min(dim=1).values

    # Extract attention from CLS token (index 0) to image patches
    # Sequence is: [CLS, Patches..., Reg1, Reg2, ...]
    num_patches = attention_heads_fused.size(-1) - 1 - num_registers

    if num_patches <= 0:
        raise ValueError(
            f"Invalid num_patches: {num_patches}. shape={attention_heads_fused.shape}, num_registers={num_registers}"
        )

    # Get CLS -> patches attention (skip CLS at index 0, and registers at the end)
    mask = attention_heads_fused[0, 0, 1:1+num_patches]

    # Reshape to 2D grid (14x14 for 224x224 images with 16x16 patches)
    width = int(np.sqrt(mask.size(-1)))
    if width * width != mask.size(-1):
        raise ValueError(f"Cannot reshape mask of size {mask.size(-1)} to square grid.")

    mask = mask.reshape(width, width).cpu().numpy()
    mask = np.maximum(mask, 0)

    # Normalize to [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask


class VITAttentionRollout:
    """
    CLS attention visualization without gradients.

    This class provides a simple visualization using only the CLS token's
    attention weights from the last layer, exactly as in the paper.
    """

    def __init__(
        self,
        model,
        head_fusion="mean",
        num_registers=0,
    ):
        self.model = model
        self.head_fusion = head_fusion
        self.num_registers = num_registers

    def __call__(self, input_tensor, category_index=None):
        """
        Compute CLS attention for the given input.

        Args:
            input_tensor: Input image tensor [batch, channels, height, width]
            category_index: Unused (kept for API compatibility)

        Returns:
            mask: CLS attention mask as numpy array [H, W]
        """
        with torch.no_grad():
            # Forward pass with attention outputs enabled
            outputs = self.model(input_tensor, output_attentions=True)
            attentions = list(outputs.attentions)

            if not attentions:
                raise RuntimeError(
                    "Model returned no attentions. Ensure output_attentions=True is supported."
                )

            # Move to CPU to reduce GPU memory pressure
            attentions_cpu = [a.detach().cpu() for a in attentions]

            return simple_cls_attention(
                attentions_cpu,
                head_fusion=self.head_fusion,
                num_registers=self.num_registers,
            )


def process_one_image(
    image_path: Path,
    output_path: Path,
    model,
    processor,
    attention_rollout,
    device,
    num_registers=0,
    enhance_mask=True,
):
    """Process a single image and save CLS attention visualization."""
    raw_img = Image.open(str(image_path)).convert("RGB")

    inputs = processor(images=raw_img, return_tensors="pt")
    input_tensor = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_idx = probs.argmax(-1).item()
        confidence = probs[0, predicted_idx].item()

    predicted_emotion = model.config.id2label[predicted_idx]

    # Compute CLS attention mask
    mask = attention_rollout(input_tensor, predicted_idx)

    img_resized = raw_img.resize((224, 224))
    original_rgb = np.array(img_resized)

    mask_resized = cv2.resize(mask, (224, 224))

    # Apply enhance_rollout_mask when requested (Mode 2)
    if enhance_mask:
        mask_resized = enhance_rollout_mask(mask_resized)

    rollout_rgb = show_mask_on_image(original_rgb, mask_resized)

    true_label = image_path.parent.name if image_path.parent != image_path.parent.parent else ""

    summary_rgb = build_summary_image(
        original_rgb=original_rgb,
        rollout_rgb=rollout_rgb,
        predicted_label=predicted_emotion,
        confidence=confidence,
        true_label=true_label,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_bgr = cv2.cvtColor(summary_rgb, cv2.COLOR_RGB2BGR)
    save_image_unicode_safe(str(output_path), summary_bgr)

    return predicted_emotion, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference with CLS attention visualization. "
                    "Supports three modes: baseline, enhanced, and with registers."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "observe"),
        help="Directory containing observe images (can include emotion subfolders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "outputs" / "observe_inference_attention"),
        help="Directory to save combined visualization images.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "models" / "emotion_vit"),
        help="Local model directory.",
    )
    parser.add_argument(
        "--head_fusion",
        type=str,
        default="mean",
        choices=["mean", "max", "min"],
        help="How to fuse attention heads. Default: mean",
    )
    parser.add_argument(
        "--use_registers",
        action="store_true",
        help="Use test-time register tokens (Mode 3)",
    )
    parser.add_argument(
        "--register_neurons",
        type=str,
        default=None,
        help="Path to register_neurons.json file",
    )
    parser.add_argument(
        "--enhance_mask",
        action="store_true",
        help="Apply enhance_rollout_mask post-processing (Mode 2). "
             "Default is False for vanilla baseline (Mode 1)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    image_paths = sorted(iter_images(input_dir))
    if not image_paths:
        print(f"No image files found in: {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Found {len(image_paths)} images in {input_dir}")

    processor = AutoImageProcessor.from_pretrained(str(model_dir))

    # Determine num_registers and load model
    num_registers = 0
    if args.use_registers:
        if args.register_neurons is None:
            register_neurons_path = model_dir.parent / "register_neurons.json"
        else:
            register_neurons_path = Path(args.register_neurons)

        if not register_neurons_path.exists():
            raise FileNotFoundError(f"Register neurons file not found: {register_neurons_path}")

        print(f"Loading model with test-time registers from {register_neurons_path}...")
        model = load_model_with_registers(
            model_path=str(model_dir),
            register_neurons_path=str(register_neurons_path),
            num_registers=1,
            device=device,
        )
        num_registers = 1
        print(f"✓ Model loaded with {num_registers} register token")
    else:
        model = ViTForImageClassification.from_pretrained(
            str(model_dir),
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()

    # Initialize CLS attention visualization (no gradients, no rollout)
    attention_rollout = VITAttentionRollout(
        model,
        head_fusion=args.head_fusion,
        num_registers=num_registers,
    )

    # Determine mode
    mode = "Mode 1 (baseline)"
    if args.use_registers:
        mode = "Mode 3 (with registers)"
    elif args.enhance_mask:
        mode = "Mode 2 (enhanced)"
    print(f"Running in {mode}")

    success = 0
    for i, image_path in enumerate(image_paths, start=1):
        rel = image_path.relative_to(input_dir)
        out_name = rel.with_suffix("").name + "_summary.png"
        out_path = output_dir / rel.parent / out_name

        try:
            # enhance_rollout_mask is only applied when --enhance_mask flag is set
            # When using registers, enhance_mask is always False (Mode 3)
            enhance_mask = args.enhance_mask and not args.use_registers
            predicted_emotion, confidence = process_one_image(
                image_path=image_path,
                output_path=out_path,
                model=model,
                processor=processor,
                attention_rollout=attention_rollout,
                device=device,
                num_registers=num_registers,
                enhance_mask=enhance_mask,
            )
            success += 1
            print(
                f"[{i}/{len(image_paths)}] OK: {image_path.name} -> {predicted_emotion} ({confidence:.2%})"
            )
        except Exception as exc:
            print(f"[{i}/{len(image_paths)}] FAIL: {image_path} | {exc}")

    print(f"Done. {success}/{len(image_paths)} images processed.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
