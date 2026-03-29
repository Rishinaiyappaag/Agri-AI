# plant_disease_predictor.py
# ============================================================
# PLANT DISEASE PREDICTOR with IMAGE VALIDATION
# ============================================================
# Rejects diagrams, illustrations, screenshots, random images.
# Accepts real photographs of plant leaves (including diseased ones
# with spots, holes, serrated edges, discoloration).
# ============================================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  opencv-python not installed! Run: pip install opencv-python")
    print("⚠️  Image validation will be LIMITED without cv2.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL THRESHOLDS
# =========================
CONFIDENCE_THRESHOLD = 85.0
ENTROPY_THRESHOLD = 0.50
TOP2_GAP_THRESHOLD = 15.0


# =========================
# LOAD MODEL
# =========================
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, class_names


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =============================================================
# IMAGE VALIDATION
# =============================================================
def validate_leaf_image(pil_image):
    """
    Validates image is a real plant leaf photograph.
    Uses a STRIKE system — image is rejected only if 3+ checks fail.
    This prevents false rejections on real diseased leaves that have
    spots, holes, serrated edges, or unusual coloring.
    """
    if not CV2_AVAILABLE:
        return True, ""

    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (224, 224))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    strikes = 0
    reasons = []

    # ─────────────────────────────────────────────
    # CHECK 1: WHITE BACKGROUND (>35%)
    # ─────────────────────────────────────────────
    # Illustrations/banners have large white areas.
    # Real leaf close-ups fill the frame.
    # Slightly relaxed: some dataset images have white paper backgrounds.
    white_ratio = np.sum(gray > 235) / gray.size
    if white_ratio > 0.35:
        strikes += 1
        reasons.append(f"Too much white background ({white_ratio:.0%}).")

    # ─────────────────────────────────────────────
    # CHECK 2: TEXTURE (Laplacian Variance)
    # ─────────────────────────────────────────────
    # Real photos: 80 - 8000 (natural texture)
    # Flat illustrations: < 60
    # Text/dense diagrams: > 8000
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 60:
        strikes += 1
        reasons.append(f"Image too smooth/flat (texture: {laplacian_var:.0f}).")
    elif laplacian_var > 8000:
        strikes += 1
        reasons.append(f"Image has dense text/lines (texture: {laplacian_var:.0f}).")

    # ─────────────────────────────────────────────
    # CHECK 3: STRAIGHT LINES (Hough Transform)
    # ─────────────────────────────────────────────
    # Diagrams/flowcharts have many straight lines.
    # Real leaves have curved, organic edges — very few straight lines.
    # Threshold: 30+ straight lines = likely a diagram.
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=50, minLineLength=30, maxLineGap=10)
    num_lines = 0 if lines is None else len(lines)
    if num_lines > 30:
        strikes += 1
        reasons.append(f"Detected {num_lines} straight lines (diagram-like).")

    # ─────────────────────────────────────────────
    # CHECK 4: COLOR HISTOGRAM — dominant flat color
    # ─────────────────────────────────────────────
    # Illustrations use flat fills → one histogram bin dominates.
    # Real photos have gradual color spread.
    hist_b = cv2.calcHist([img_resized], [0], None, [64], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [64], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_resized], [2], None, [64], [0, 256]).flatten()
    total_px = 224 * 224
    max_bin = max(hist_b.max(), hist_g.max(), hist_r.max()) / total_px
    if max_bin > 0.30:
        strikes += 1
        reasons.append(f"Unnaturally dominant color block ({max_bin:.0%}).")

    # ─────────────────────────────────────────────
    # CHECK 5: UNIQUE COLORS
    # ─────────────────────────────────────────────
    # Real photos: many gradual transitions → many unique colors.
    # Flat illustrations: few solid fills → few unique colors.
    small = cv2.resize(img_resized, (32, 32))
    small_q = (small // 24) * 24
    unique_colors = len(np.unique(small_q.reshape(-1, 3), axis=0))
    if unique_colors < 30:
        strikes += 1
        reasons.append(f"Only {unique_colors} color regions (illustration-like).")

    # ─────────────────────────────────────────────
    # CHECK 6: SATURATION
    # ─────────────────────────────────────────────
    mean_sat = np.mean(hsv[:, :, 1])
    if mean_sat < 15:
        strikes += 1
        reasons.append(f"Image too desaturated ({mean_sat:.0f}).")
    elif mean_sat > 210:
        strikes += 1
        reasons.append(f"Colors unnaturally vivid ({mean_sat:.0f}).")

    # ─────────────────────────────────────────────
    # CHECK 7: PLANT COLOR PRESENCE
    # ─────────────────────────────────────────────
    # Leaves are green, yellow-green, brown, or yellow.
    # Images without these colors are not plant images.
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green_mask = (h >= 25) & (h <= 95) & (s >= 25) & (v >= 25)
    yb_mask = (h >= 8) & (h <= 30) & (s >= 25) & (v >= 25)
    plant_ratio = np.sum(green_mask | yb_mask) / h.size
    if plant_ratio < 0.08:
        strikes += 1
        reasons.append(f"Not enough plant colors ({plant_ratio:.0%}).")

    # ─────────────────────────────────────────────
    # DECISION: Reject if 3 or more strikes
    # ─────────────────────────────────────────────
    # Why 3? Because real diseased leaves can trigger 1-2 checks:
    #   - Serrated edges → slightly high edge count
    #   - Leaf on white paper → white background check
    #   - Dark lesions → low saturation in spots
    # But they won't trigger 3+ checks simultaneously.
    # Illustrations/diagrams typically fail 4-6 checks.
    if strikes >= 3:
        msg = (
            "This image does not appear to be a plant leaf photograph. "
            "Please upload a clear, close-up photo of a real plant leaf."
        )
        return False, msg

    return True, ""


# =============================================================
# CORE PREDICTION
# =============================================================
def _predict_from_tensor(image_tensor, model, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    raw_conf = confidence.item() * 100

    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).item()
    max_entropy = np.log(len(class_names))
    normalized_entropy = entropy / max_entropy

    top2_probs, _ = torch.topk(probs, 2, dim=1)
    top1 = top2_probs[0][0].item() * 100
    top2 = top2_probs[0][1].item() * 100
    top2_gap = top1 - top2

    is_rejected = (
        raw_conf < CONFIDENCE_THRESHOLD or
        normalized_entropy > ENTROPY_THRESHOLD or
        top2_gap < TOP2_GAP_THRESHOLD
    )

    if is_rejected:
        return {
            "disease": "Unknown",
            "confidence": round(raw_conf, 2),
            "is_valid": False,
            "message": "The model could not confidently identify a plant disease. "
                       "Please upload a clear, close-up photo of a single plant leaf."
        }

    return {
        "disease": class_names[predicted.item()],
        "confidence": round(raw_conf, 2),
        "is_valid": True,
        "message": ""
    }


# =============================================================
# PUBLIC API
# =============================================================
def predict(image_path, model, class_names):
    """Predict from file path."""
    pil_image = Image.open(image_path).convert("RGB")

    is_leaf, reason = validate_leaf_image(pil_image)
    if not is_leaf:
        return {
            "disease": "Unknown",
            "confidence": 0.0,
            "is_valid": False,
            "message": reason
        }

    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    return _predict_from_tensor(image_tensor, model, class_names)


def predict_from_pil(pil_image, model, class_names):
    """Predict from PIL Image (camera capture)."""
    pil_image = pil_image.convert("RGB")

    is_leaf, reason = validate_leaf_image(pil_image)
    if not is_leaf:
        return {
            "disease": "Unknown",
            "confidence": 0.0,
            "is_valid": False,
            "message": reason
        }

    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    return _predict_from_tensor(image_tensor, model, class_names)