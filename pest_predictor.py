# pest_predictor.py
# ============================================================
# PEST PREDICTOR — works with MobileNetV2 retrained model
# ============================================================
# Supports both old model (64x64) and new model (128x128).
# Loads class names from pest_class_names.json if available,
# falls back to hardcoded PEST_MAP otherwise.
# ============================================================

import numpy as np
from PIL import Image
import json
import os

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  opencv-python not installed for pest validation!")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  tensorflow not available for pest prediction!")


# =========================
# LOAD CONFIG
# =========================
# Try to load class names and image size from JSON (saved during training)
PEST_CLASS_NAMES_FILE = "pest_class_names.json"

if os.path.exists(PEST_CLASS_NAMES_FILE):
    with open(PEST_CLASS_NAMES_FILE, "r") as f:
        pest_config = json.load(f)
    CLASS_NAMES = pest_config["class_names"]
    IMG_SIZE = pest_config.get("img_size", 128)
    NUM_CLASSES = pest_config.get("num_classes", len(CLASS_NAMES))
    print(f"  ✓ Pest config loaded: {NUM_CLASSES} classes, {IMG_SIZE}x{IMG_SIZE}")
else:
    # Fallback to hardcoded values (for old 64x64 model)
    CLASS_NAMES = [
        "aphids", "armyworm", "beetle", "bollworm", "earthworm",
        "grasshopper", "mites", "mosquito", "sawfly", "stem borer"
    ]
    IMG_SIZE = 64
    NUM_CLASSES = 10
    print(f"  ⚠️  pest_class_names.json not found, using defaults (64x64)")

# Build index-to-name map
PEST_MAP = {i: name for i, name in enumerate(CLASS_NAMES)}

# =========================
# THRESHOLDS
# =========================
CONFIDENCE_THRESHOLD = 85.0
TOP2_GAP_THRESHOLD = 25.0
ENTROPY_THRESHOLD = 0.5


# =============================================================
# IMAGE VALIDATION
# =============================================================
def validate_pest_image(pil_image):
    if not CV2_AVAILABLE:
        return True, ""

    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (224, 224))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    strikes = 0

    white_ratio = np.sum(gray > 235) / gray.size
    if white_ratio > 0.35:
        strikes += 1

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 60 or laplacian_var > 8000:
        strikes += 1

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None and len(lines) > 30:
        strikes += 1

    hist_b = cv2.calcHist([img_resized], [0], None, [64], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [64], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_resized], [2], None, [64], [0, 256]).flatten()
    total_px = 224 * 224
    max_bin = max(hist_b.max(), hist_g.max(), hist_r.max()) / total_px
    if max_bin > 0.30:
        strikes += 1

    small = cv2.resize(img_resized, (32, 32))
    small_q = (small // 24) * 24
    unique_colors = len(np.unique(small_q.reshape(-1, 3), axis=0))
    if unique_colors < 30:
        strikes += 1

    mean_sat = np.mean(hsv[:, :, 1])
    if mean_sat < 15 or mean_sat > 210:
        strikes += 1

    if strikes >= 3:
        return False, (
            "This image does not appear to be a photograph of a pest/insect. "
            "Please upload a clear, close-up photo of the pest."
        )

    return True, ""


# =============================================================
# MODEL-LEVEL CONFIDENCE CHECK
# =============================================================
def _check_model_confidence(preds):
    probs = preds[0]

    sorted_probs = np.sort(probs)[::-1]
    top1_conf = sorted_probs[0] * 100
    top2_conf = sorted_probs[1] * 100
    top2_gap = top1_conf - top2_conf

    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(NUM_CLASSES)
    normalized_entropy = entropy / max_entropy

    if top1_conf < CONFIDENCE_THRESHOLD:
        return False, (
            f"Model confidence too low ({top1_conf:.1f}%). "
            f"Please upload a clear, close-up photo of a pest."
        )

    if top2_gap < TOP2_GAP_THRESHOLD:
        return False, (
            f"Model is uncertain between multiple classes. "
            f"This image may not contain a recognizable pest."
        )

    if normalized_entropy > ENTROPY_THRESHOLD:
        return False, (
            f"Model predictions are too spread out. "
            f"This image does not appear to contain a recognizable pest."
        )

    return True, ""


# =============================================================
# CORE PREDICTION
# =============================================================
def _predict_from_pil(pil_image, model):
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)

    is_confident, reason = _check_model_confidence(preds)
    if not is_confident:
        return {
            "pest_name": "unknown",
            "confidence": round(float(np.max(preds)) * 100, 2),
            "is_valid": False,
            "message": reason
        }

    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    pest_name = PEST_MAP.get(pred_index, "unknown")

    return {
        "pest_name": pest_name,
        "confidence": round(confidence, 2),
        "is_valid": True,
        "message": ""
    }


# =============================================================
# PUBLIC API
# =============================================================
def predict_pest(img_path, model):
    pil_image = Image.open(img_path).convert("RGB")

    is_valid, reason = validate_pest_image(pil_image)
    if not is_valid:
        return {
            "pest_name": "unknown",
            "confidence": 0.0,
            "is_valid": False,
            "message": reason
        }

    return _predict_from_pil(pil_image, model)


def predict_pest_from_pil(pil_image, model):
    pil_image = pil_image.convert("RGB")

    is_valid, reason = validate_pest_image(pil_image)
    if not is_valid:
        return {
            "pest_name": "unknown",
            "confidence": 0.0,
            "is_valid": False,
            "message": reason
        }

    return _predict_from_pil(pil_image, model)