"""
Utility functions for emotion recognition inference.

Includes face detection, visualization helpers, and attention mask processing.
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image


def _parse_retinaface_bbox(face_data):
    """Extract [x1, y1, x2, y2] bbox from RetinaFace response entry."""
    if isinstance(face_data, dict):
        if "facial_area" in face_data and len(face_data["facial_area"]) >= 4:
            box = face_data["facial_area"]
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])
        if "bbox" in face_data and len(face_data["bbox"]) >= 4:
            box = face_data["bbox"]
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if isinstance(face_data, (list, tuple)) and len(face_data) >= 4:
        return int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
    return None


def detect_and_crop_faces(image_rgb, expand_ratio=0.3):
    """Detect faces with RetinaFace and return expanded crops.

    Returns:
        detections: list of dicts with keys raw_bbox and expanded_bbox
        crops_rgb: list of np.ndarray, one expanded crop per detection
    """
    if not (0.0 <= expand_ratio <= 1.0):
        raise ValueError("expand_ratio must be between 0 and 1.")

    from retinaface import RetinaFace

    img = np.asarray(image_rgb, dtype=np.uint8)
    h, w = img.shape[:2]
    resp = RetinaFace.detect_faces(img)

    if not isinstance(resp, dict) or len(resp) == 0:
        return [], []

    detections = []
    crops_rgb = []
    for _, face_data in resp.items():
        parsed = _parse_retinaface_bbox(face_data)
        if parsed is None:
            continue
        x1, y1, x2, y2 = parsed

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        expand_w = int(bw * expand_ratio)
        expand_h = int(bh * expand_ratio)

        ex1 = max(0, x1 - expand_w)
        ey1 = max(0, y1 - expand_h)
        ex2 = min(w, x2 + expand_w)
        ey2 = min(h, y2 + expand_h)

        if ex2 <= x1 or ey2 <= y1:
            continue

        crop = img[ey1:ey2, ex1:ex2].copy()
        detections.append(
            {
                "raw_bbox": (x1, y1, x2, y2),
                "expanded_bbox": (ex1, ey1, ex2, ey2),
            }
        )
        crops_rgb.append(crop)

    return detections, crops_rgb


def draw_face_boxes(image_rgb, detections):
    """Draw RetinaFace raw and expanded boxes for visualization."""
    canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["raw_bbox"]
        ex1, ey1, ex2, ey2 = det["expanded_bbox"]
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (255, 196, 0), 2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"Face {idx}",
            (ex1, max(12, ey1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 196, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def enhance_rollout_mask(mask):
    """
    Improve attention mask quality by discarding extreme spikes and
    repairing discarded holes using local median filtering.
    """
    low_percent = 20  # Set lowest 20% to zero
    local_d = 21
    outlier_z = 6.0
    fill_ksize = int(local_d) if int(local_d) % 2 == 1 else int(local_d) + 1
    fill_ksize = max(fill_ksize, 3)

    mask = np.asarray(mask, dtype=np.float32)
    mask = np.clip(mask, 0.0, None)

    # Set lowest X% to zero
    low_thresh = np.percentile(mask, low_percent)
    mask[mask <= low_thresh] = 0.0

    median = np.median(mask)
    mad = np.median(np.abs(mask - median))
    robust_sigma = 1.4826 * mad

    if robust_sigma < 1e-8:
        return np.clip(mask / (mask.max() + 1e-8), 0.0, 1.0)

    # Value-based discard: values above upper_cap are set to zero
    upper_cap = median + outlier_z * robust_sigma
    discarded = mask > upper_cap
    mask = np.where(discarded, 0.0, mask)
    mask = mask / (upper_cap + 1e-8)

    # Expand discarded points to their local_d neighborhoods, then median-filter
    if discarded.any():
        mask_uint8 = np.uint8(np.clip(mask * 255, 0, 255))
        median_filtered = cv2.medianBlur(mask_uint8, fill_ksize).astype(np.float32) / 255.0

        kernel = np.ones((fill_ksize, fill_ksize), dtype=np.uint8)
        repair_region = cv2.dilate(discarded.astype(np.uint8), kernel, iterations=1).astype(bool)

        mask = np.where(repair_region, median_filtered, mask)

    return np.clip(mask, 0.0, 1.0)


def show_mask_on_image(img, mask):
    """Overlay attention heatmap on original image."""
    img = np.asarray(img, dtype=np.float32) / 255.0
    mask = np.asarray(mask, dtype=np.float32)
    mask = np.clip(mask, 0.0, 1.0)

    # OpenCV colormaps are BGR; convert to RGB
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Blend using mask as per-pixel alpha
    alpha = np.expand_dims(mask, axis=-1)
    cam = img * (1.0 - alpha) + heatmap * alpha
    return np.uint8(np.clip(cam, 0.0, 1.0) * 255)


def save_image_unicode_safe(path, image_bgr):
    """Save an image robustly even when the path contains non-ASCII characters."""
    ok = cv2.imwrite(path, image_bgr)
    if ok:
        return

    ext = os.path.splitext(path)[1] or ".png"
    success, encoded = cv2.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError(f"Failed to encode image for '{path}'.")

    # OpenCV can fail on some Windows Unicode paths; write encoded bytes directly
    encoded.tofile(path)
    if not os.path.exists(path):
        raise RuntimeError(f"Failed to save visualization to '{path}'.")
