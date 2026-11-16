
import cv2
import numpy as np


def _odd_kernel(value, minimum=3, maximum=31):
    """Ensure an odd kernel size."""
    value = max(minimum, min(int(value), maximum))

    if value % 2 == 0:
        value += 1

    return min(value, maximum)


def _smooth_colors(img, passes=2):
    """Smooth image colours while preserving boundaries."""
    result = img.copy()

    for _ in range(passes):
        result = cv2.bilateralFilter(
            result,
            9,
            100,
            100
        )

    return result


def _clean_edges(img, threshold1=50, threshold2=130):
    """Generate clean cartoon outlines."""
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.bilateralFilter(
        gray,
        7,
        60,
        60
    )

    edges = cv2.Canny(
        gray,
        threshold1,
        threshold2
    )

    # Remove isolated edge noise
    kernel = np.ones((2, 2), np.uint8)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    return edges


def _quantize_colors(img, k=8):
    """
    Reduce the image to a limited colour palette
    using K-Means clustering.
    """

    height, width = img.shape[:2]

    # Reduce image size for faster and more stable clustering
    small = cv2.resize(
        img,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA
    )

    data = np.float32(
        small.reshape((-1, 3))
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.5
    )

    compactness, labels, centers = cv2.kmeans(
        data,
        int(k),
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    quantized = centers[
        labels.flatten()
    ].reshape(small.shape)

    # Restore original image dimensions
    quantized = cv2.resize(
        quantized,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    return quantized


def cartoonize(
    img,
    d=9,
    sigma_color=75,
    sigma_space=75,
    block_size=9,
    C=2
):
    """
    Classic cartoon effect.

    Produces smooth colour regions with clear dark outlines.
    """

    d = max(5, int(d))
    block_size = _odd_kernel(block_size, 3, 31)

    # Smooth the image to remove photographic texture
    color = img.copy()

    for _ in range(3):
        color = cv2.bilateralFilter(
            color,
            d,
            sigma_color,
            sigma_space
        )

    # Simplify colours slightly
    color = _quantize_colors(
        color,
        k=12
    )

    color = cv2.bilateralFilter(
        color,
        9,
        100,
        100
    )

    # Generate clean outlines
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.medianBlur(
        gray,
        5
    )

    # Use adaptive thresholding with less sensitivity
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # Clean tiny isolated marks
    kernel = np.ones((2, 2), np.uint8)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Combine colour regions and outlines
    cartoon = cv2.bitwise_and(
        color,
        color,
        mask=edges
    )

    return cartoon


def cartoonize_kmeans(img, k=8, d=9, sigma=80):
    """
    K-Means cartoon effect.

    Creates a poster-like illustration with larger,
    simplified colour regions and clean outlines.
    """

    k = max(3, min(int(k), 16))

    # First smooth the photograph
    smoothed = _smooth_colors(
        img,
        passes=2
    )

    # Quantize the colours
    quantized = _quantize_colors(
        smoothed,
        k=k
    )

    # Smooth boundaries between colour regions
    quantized = cv2.bilateralFilter(
        quantized,
        max(5, int(d)),
        sigma,
        sigma
    )

    # Generate clean outlines from the original image
    edges = _clean_edges(
        img,
        threshold1=45,
        threshold2=120
    )

    # Convert edge mask to black outlines
    edge_mask = 255 - edges

    # Slightly soften the outlines
    edge_mask = cv2.GaussianBlur(
        edge_mask,
        (3, 3),
        0
    )

    edge_mask = cv2.cvtColor(
        edge_mask,
        cv2.COLOR_GRAY2BGR
    )

    # Combine quantized colours with outlines
    cartoon = cv2.multiply(
        quantized,
        edge_mask,
        scale=1 / 255.0
    )

    return cartoon


def oil_painting_effect(img, size=7, dynRatio=1):
    """
    Oil painting effect.

    Uses OpenCV's xphoto oil painting algorithm when available.
    Otherwise uses OpenCV stylization or a painterly fallback.
    """

    size = max(3, min(int(size), 15))

    if size % 2 == 0:
        size += 1

    # Preferred implementation: OpenCV contrib
    if hasattr(cv2, "xphoto"):

        try:
            return cv2.xphoto.oilPainting(
                img,
                size,
                int(dynRatio)
            )

        except (cv2.error, AttributeError):
            pass

    # Second option: OpenCV's built-in stylization
    if hasattr(cv2, "stylization"):

        try:
            result = cv2.stylization(
                img,
                sigma_s=60,
                sigma_r=0.45
            )

            return result

        except cv2.error:
            pass

    # Fallback: smooth, quantize and enhance paint regions
    smoothed = cv2.bilateralFilter(
        img,
        9,
        100,
        100
    )

    quantized = _quantize_colors(
        smoothed,
        k=16
    )

    result = cv2.medianBlur(
        quantized,
        5
    )

    return result
