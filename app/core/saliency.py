# ============================================
# core/saliency.py
# ============================================
import cv2
import numpy as np

def frequency_tuned_saliency(image):
    """
    Simplified implementation of the FTS method (Achanta et al., 2009).
    Computes a saliency map based on the pixel-wise deviation from the global mean color in Lab space.
    """
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    mean_color = cv2.mean(img_lab)[:3]

    # Apply Gaussian smoothing to reduce noise before computing differences
    blurred = cv2.GaussianBlur(img_lab, (5, 5), 0)

    # Saliency is defined as the Euclidean distance from the mean Lab color
    saliency_map = np.linalg.norm(blurred - mean_color, axis=2)

    # Normalize saliency to a [0, 1] range for consistency
    saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
    return saliency_map


def graph_based_saliency(image):
    """
    Lightweight approximation of GBVS (Graph-Based Visual Saliency).
    True GBVS involves complex graph computations; this approximation captures 
    the core idea by mixing local contrast and luminance-based gradients.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Slight blur helps suppress high-frequency noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Laplacian highlights strong intensity transitions (edges/regions of contrast)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    saliency_map = np.abs(laplacian)

    # Normalize to [0, 1] for compatibility with other saliency methods
    saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
    return saliency_map


def compute_combined_saliency(image):
    """
    Combines GBVS-style and FTS saliency maps by simple averaging,
    following the blending approach discussed in the original papers.
    """
    s1 = graph_based_saliency(image)
    s2 = frequency_tuned_saliency(image)

    # Average the two maps to retain both global and local saliency cues
    combined = (s1 + s2) / 2.0

    # Apply final smoothing for a softer, more coherent saliency distribution
    combined = cv2.GaussianBlur(combined, (5, 5), 0)

    # Normalize to ensure a clean [0, 1] output range
    combined = cv2.normalize(combined, None, 0, 1, cv2.NORM_MINMAX)
    return combined
