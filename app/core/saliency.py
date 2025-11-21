# ============================================
# core/saliency.py
# ============================================
import cv2
import numpy as np
from app.core.log_utils import get_logger

logger = get_logger(__name__)

def frequency_tuned_saliency(image):
    """
    Simplified implementation of the FTS method (Achanta et al., 2009).
    Computes a saliency map based on the pixel-wise deviation from the global mean color in Lab space.
    """
    logger.debug("Computing FTS saliency")
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    mean_color = cv2.mean(img_lab)[:3]

    blurred = cv2.GaussianBlur(img_lab, (5, 5), 0)

    saliency_map = np.linalg.norm(blurred - mean_color, axis=2)

    saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
    
    logger.debug(f"FTS saliency computed - min: {saliency_map.min():.3f}, max: {saliency_map.max():.3f}")
    return saliency_map


def graph_based_saliency(image):
    """
    Lightweight approximation of GBVS (Graph-Based Visual Saliency).
    True GBVS involves complex graph computations; this approximation captures 
    the core idea by mixing local contrast and luminance-based gradients.
    """
    logger.debug("Computing GBVS saliency")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    saliency_map = np.abs(laplacian)

    saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
    
    logger.debug(f"GBVS saliency computed - min: {saliency_map.min():.3f}, max: {saliency_map.max():.3f}")
    return saliency_map


def compute_combined_saliency(image):
    """
    Combines GBVS-style and FTS saliency maps by simple averaging,
    following the blending approach discussed in the original papers.
    """
    logger.info("Starting combined saliency computation")
    s1 = graph_based_saliency(image)
    s2 = frequency_tuned_saliency(image)

    combined = (s1 + s2) / 2.0

    combined = cv2.GaussianBlur(combined, (5, 5), 0)

    combined = cv2.normalize(combined, None, 0, 1, cv2.NORM_MINMAX)
    
    logger.info(f"Combined saliency computed - min: {combined.min():.3f}, max: {combined.max():.3f}, mean: {combined.mean():.3f}")
    return combined