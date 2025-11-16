# ============================================
# core/utils.py
# ============================================
import cv2
import numpy as np
from skimage import color
import json
import os

def rgb_to_lab(image):
    """
    Converts an RGB image into the Lab color space.
    """
    # Direct RGB → Lab conversion using OpenCV,
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Convert OpenCV’s Lab ranges into standard CIE Lab ranges:
    # OpenCV: L* [0,255], a* [0,255], b* [0,255]
    # Standard: L* [0,100], a* [-128,127], b* [-128,127]
    lab_standard = lab_image.astype(np.float32)
    lab_standard[:, :, 0] = lab_standard[:, :, 0] * 100.0 / 255.0  # L* → [0,100]
    lab_standard[:, :, 1] = lab_standard[:, :, 1] - 128.0          # a* shift to [-128,127]
    lab_standard[:, :, 2] = lab_standard[:, :, 2] - 128.0          # b* shift to [-128,127]
    
    return lab_standard

def lab_to_rgb(image_lab):
    """
    Converts a Lab image back to the RGB color space.

    """
    lab_opencv = image_lab.astype(np.float32).copy()
    lab_opencv[:, :, 0] = lab_opencv[:, :, 0] * 255.0 / 100.0      # L* → [0,255]
    lab_opencv[:, :, 1] = lab_opencv[:, :, 1] + 128.0              # a* → [0,255]
    lab_opencv[:, :, 2] = lab_opencv[:, :, 2] + 128.0              # b* → [0,255]
    
    # Ensure valid 8-bit pixel values
    lab_opencv = np.clip(lab_opencv, 0, 255).astype(np.uint8)
    
    # Convert Lab → RGB using OpenCV
    rgb_image = cv2.cvtColor(lab_opencv, cv2.COLOR_LAB2RGB)
    
    return rgb_image

def normalize_map(saliency_map):
    """
    Normalizes a saliency map into the range [0,1].
    """
    # Avoid division by zero when the map has no variation
    if saliency_map.max() == saliency_map.min():
        return np.ones_like(saliency_map) * 0.5
    
    saliency_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    return saliency_normalized

def save_palette(palette, path="data/palettes/palette.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(palette, f, indent=4)

def rgb_to_hex(color):
    """
    Converts an RGB color to a HEX string.
    """
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def debug_saliency_distribution(saliency_map):
    """
    Returns descriptive statistics about a saliency map.
    Useful for debugging saliency behavior on different images.
    """
    stats = {
        'min': float(saliency_map.min()),
        'max': float(saliency_map.max()),
        'mean': float(saliency_map.mean()),
        'std': float(saliency_map.std()),
        'percentile_10': float(np.percentile(saliency_map, 10)),
        'percentile_90': float(np.percentile(saliency_map, 90))
    }
    return stats
