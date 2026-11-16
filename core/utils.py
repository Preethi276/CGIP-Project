import cv2
import numpy as np


def resize_keep_aspect(img, max_width=900, max_height=700):
    """Resize image while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def side_by_side(img1, img2, gap=12):
    """Place two images side by side with a small gap."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)

    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    canvas = np.ones((h, w1 + w2 + gap, 3), dtype=np.uint8) * 240
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 + gap:] = img2
    return canvas


def apply_clahe(gray):
    """Improve contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
