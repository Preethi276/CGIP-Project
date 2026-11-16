
import cv2
import numpy as np


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _odd(value):
    """Ensure that the Gaussian kernel size is odd."""
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _graphite_shading(img, blur_ksize=21):
    """
    Generate a grayscale graphite shading layer
    using the dodge blending technique.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Remove small photographic noise while preserving details
    gray = cv2.bilateralFilter(
        gray,
        9,
        45,
        45
    )

    # Gentle local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=1.35,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Dodge blending
    kernel = _odd(max(blur_ksize, 31))

    inverted = 255 - gray

    blurred = cv2.GaussianBlur(
        inverted,
        (kernel, kernel),
        0
    )

    denominator = np.maximum(
        255 - blurred,
        1
    )

    dodge = (
        gray.astype(np.float32) * 255.0
        / denominator.astype(np.float32)
    )

    dodge = np.clip(
        dodge,
        0,
        255
    ).astype(np.uint8)

    return gray, dodge


# =========================================================
# 1. BLACK AND WHITE PENCIL SKETCH
# =========================================================

def pencil_sketch(img, blur_ksize=21, enhance=True):
    """
    Graphite pencil sketch.

    Features:
    - Smooth graphite shading
    - Darker pencil appearance
    - Improved facial detail preservation
    - Controlled contrast
    - Subtle pencil contours

    Parameters:
        img: Input BGR image
        blur_ksize: Gaussian blur kernel size
        enhance: Retained for compatibility with existing code

    Returns:
        Grayscale pencil sketch
    """

    if img is None or img.size == 0:
        raise ValueError("Input image is empty.")

    gray, dodge = _graphite_shading(
        img,
        blur_ksize
    )

    # Blend dodge shading with original grayscale information
    result = cv2.addWeighted(
        dodge,
        0.62,
        gray,
        0.38,
        0
    )

    # Improve graphite contrast
    result = cv2.convertScaleAbs(
        result,
        alpha=1.18,
        beta=-25
    )

    # Extract subtle contours
    smooth = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        smooth,
        45,
        110
    )

    edges = cv2.GaussianBlur(
        edges,
        (3, 3),
        0
    )

    # Apply gentle edge definition
    edge_strength = (
        edges.astype(np.float32) / 255.0
    )

    result = (
        result.astype(np.float32)
        * (1.0 - 0.12 * edge_strength)
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# =========================================================
# 2. COLOUR PENCIL SKETCH
# =========================================================

def color_pencil_sketch(img, blur_ksize=21):
    """
    Coloured pencil illustration.

    Features:
    - Simplified colour regions
    - Pencil-style tonal shading
    - Natural colour preservation
    - Fine contours
    - Increased colour intensity

    Parameters:
        img: Input BGR image
        blur_ksize: Gaussian blur kernel size

    Returns:
        Coloured BGR pencil illustration
    """

    if img is None or img.size == 0:
        raise ValueError("Input image is empty.")

    gray, dodge = _graphite_shading(
        img,
        blur_ksize
    )

    # -----------------------------------------------------
    # STEP 1: Smooth and simplify colours
    # -----------------------------------------------------

    color = cv2.bilateralFilter(
        img,
        11,
        80,
        80
    )

    color = cv2.bilateralFilter(
        color,
        9,
        55,
        55
    )

    # Mean-shift filtering simplifies colour regions
    simplified = cv2.pyrMeanShiftFiltering(
        color,
        sp=10,
        sr=18,
        maxLevel=1
    )

    # Retain some original colour information
    color = cv2.addWeighted(
        simplified,
        0.72,
        color,
        0.28,
        0
    )

    # -----------------------------------------------------
    # STEP 2: Create pencil shading
    # -----------------------------------------------------

    tonal = dodge.astype(np.float32) / 255.0

    tonal = np.clip(
        tonal,
        0.0,
        1.0
    )

    shading = 0.48 + 0.52 * tonal

    # Apply shading to colour channels
    result = (
        color.astype(np.float32)
        * shading[:, :, None]
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    # -----------------------------------------------------
    # STEP 3: Extract fine pencil contours
    # -----------------------------------------------------

    smooth_gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        smooth_gray,
        40,
        105
    )

    edges = cv2.GaussianBlur(
        edges,
        (3, 3),
        0
    )

    edge_strength = (
        edges.astype(np.float32) / 255.0
    )

    # Soft pencil contours instead of thick black outlines
    result = (
        result.astype(np.float32)
        * (
            1.0
            - 0.22 * edge_strength[:, :, None]
        )
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    # -----------------------------------------------------
    # STEP 4: Improve colour appearance
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * 1.18,
        0,
        255
    ).astype(np.uint8)

    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return result
