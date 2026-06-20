"""
Flask backend for emotion recognition with attention visualization.
"""

import os
import base64
import shutil
import threading
from collections import Counter
from functools import lru_cache
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, ViTForImageClassification

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import (
    detect_and_crop_faces,
    draw_face_boxes,
    enhance_rollout_mask,
    show_mask_on_image,
)
from vit_with_registers import load_model_with_registers


app = Flask(__name__)

# Guard model execution to avoid race conditions / OOM under heavy concurrent traffic.
INFERENCE_LOCK = threading.Lock()


def _prepare_retinaface_local_weights():
    """Prepare RetinaFace weights from local copy to avoid re-downloading."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    project_weight = os.path.join(project_root, "models", "retinaface", "retinaface.h5")

    # DeepFace internally appends ".deepface" to DEEPFACE_HOME.
    deepface_home = os.getenv("DEEPFACE_HOME", project_root)
    if os.path.basename(os.path.normpath(deepface_home)) == ".deepface":
        deepface_home = os.path.dirname(os.path.normpath(deepface_home))

    os.environ["DEEPFACE_HOME"] = deepface_home
    runtime_weight = os.path.join(deepface_home, ".deepface", "weights", "retinaface.h5")

    if os.path.exists(project_weight) and not os.path.exists(runtime_weight):
        os.makedirs(os.path.dirname(runtime_weight), exist_ok=True)
        shutil.copy2(project_weight, runtime_weight)

    return project_weight, runtime_weight

# Mode definitions
MODE_VANILLA = "vanilla"
MODE_ENHANCED = "enhanced"
MODE_REGISTERS = "registers"


@lru_cache(maxsize=1)
def load_runtime(mode=MODE_VANILLA):
    """Load model and processor (cached)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_model_path = os.path.join(project_root, "models", "emotion_vit")
    register_neurons_path = os.path.join(project_root, "models", "register_neurons.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processor from local directory only (no download)
    processor = AutoImageProcessor.from_pretrained(local_model_path, local_files_only=True)

    if mode == MODE_REGISTERS:
        if not os.path.exists(register_neurons_path):
            raise FileNotFoundError(f"Register neurons file not found: {register_neurons_path}")

        model = load_model_with_registers(
            model_path=local_model_path,
            register_neurons_path=register_neurons_path,
            num_registers=1,
            device=device,
        )
    else:
        model = ViTForImageClassification.from_pretrained(
            local_model_path,
            attn_implementation="eager",
            local_files_only=True,
        )
        model.to(device)
        model.eval()

    return device, processor, model


def simple_cls_attention(model, input_tensor, num_registers=0):
    """Extract CLS token attention from the last layer."""
    with torch.no_grad():
        outputs = model(input_tensor, output_attentions=True)
        attentions = list(outputs.attentions)

    # Use attention from the last layer only
    attention = attentions[-1]

    # Fuse attention heads (mean)
    attention_heads_fused = attention.mean(dim=1)

    # Extract attention from CLS token (index 0) to image patches
    num_patches = attention_heads_fused.size(-1) - 1 - num_registers

    if num_patches <= 0:
        raise ValueError(f"Invalid num_patches: {num_patches}")

    mask = attention_heads_fused[0, 0, 1:1+num_patches]

    # Reshape to 2D grid
    width = int(np.sqrt(mask.size(-1)))
    if width * width != mask.size(-1):
        raise ValueError(f"Cannot reshape mask to square grid")

    mask = mask.reshape(width, width).cpu().numpy()
    mask = np.maximum(mask, 0)

    # Normalize to [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask


def image_to_base64(img: Image.Image | np.ndarray, format: str = "PNG") -> str:
    """Convert PIL Image or numpy array to base64 string."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def run_inference(image_bytes: bytes, mode: str = MODE_VANILLA):
    """Run inference on image and return results."""
    # Prepare RetinaFace local weights
    _prepare_retinaface_local_weights()

    # Load image
    raw_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(raw_img)

    # Load model
    device, processor, model = load_runtime(mode=mode)
    num_registers = 1 if mode == MODE_REGISTERS else 0
    enhance_mask = (mode == MODE_ENHANCED)

    # Detect faces
    detections, crops_rgb = detect_and_crop_faces(raw_img, expand_ratio=0.3)

    if not detections:
        boxed_image = img_array
        return {
            "success": True,
            "faces_detected": 0,
            "boxed_image": image_to_base64(boxed_image),
            "faces": [],
            "summary": "No face detected in the input image."
        }

    # Draw bounding boxes
    boxed_image = draw_face_boxes(raw_img, detections)

    # Process each face
    faces = []
    emotion_counter = Counter()

    with INFERENCE_LOCK:
        for idx, crop_rgb in enumerate(crops_rgb, start=1):
            face_img = Image.fromarray(crop_rgb)
            inputs = processor(images=face_img, return_tensors="pt")
            input_tensor = inputs["pixel_values"].to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[0, predicted_idx].item()

            predicted_emotion = model.config.id2label[predicted_idx]
            emotion_counter[predicted_emotion] += 1

            # Generate CLS attention mask
            mask = simple_cls_attention(model, input_tensor, num_registers=num_registers)

            h, w = crop_rgb.shape[:2]
            mask_resized = np.array(
                Image.fromarray(np.uint8(mask * 255)).resize((w, h), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0

            # Apply enhance_rollout_mask only for enhanced mode
            if enhance_mask:
                mask_resized = enhance_rollout_mask(mask_resized)

            attention_rgb = show_mask_on_image(crop_rgb, mask_resized)

            faces.append({
                "index": idx,
                "crop": image_to_base64(crop_rgb),
                "attention": image_to_base64(attention_rgb),
                "emotion": predicted_emotion,
                "confidence": f"{confidence:.2%}"
            })

    # Build summary
    total_faces = len(crops_rgb)
    summary_lines = [f"Detected faces: {total_faces}", "", "Emotion distribution:"]
    for emotion, count in emotion_counter.most_common():
        pct = (count / total_faces) * 100.0
        summary_lines.append(f"- {emotion.upper()}: {count}/{total_faces} ({pct:.1f}%)")

    return {
        "success": True,
        "faces_detected": total_faces,
        "boxed_image": image_to_base64(boxed_image),
        "faces": faces,
        "summary": "\n".join(summary_lines)
    }


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/inference", methods=["POST"])
def api_inference():
    """Handle inference requests."""
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data["image"]
        mode = data.get("mode", MODE_VANILLA)

        # Decode base64 image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        # Run inference
        result = run_inference(image_bytes, mode)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Pre-load vanilla model on startup
    print("Pre-loading vanilla model...")
    try:
        load_runtime(MODE_VANILLA)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")

    app.run(host="0.0.0.0", port=7860, debug=True)
