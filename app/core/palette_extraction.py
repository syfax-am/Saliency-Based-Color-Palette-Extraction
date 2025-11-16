# ============================================
# core/palette_extraction.py
# ============================================
import numpy as np
import cv2
from app.core.distance_metrics import weighted_lab_distance
from app.core.utils import rgb_to_lab, lab_to_rgb

def extract_palette(image, saliency_map):

    # Convert the input image to CIELab for perceptual uniformity
    lab_image = rgb_to_lab(image)
    h, w, _ = lab_image.shape

    # Flatten pixel values and their associated saliency scores
    pixels = lab_image.reshape(-1, 3)
    saliency_flat = saliency_map.flatten()

    # Adaptive filtering using percentile-based thresholds
    #This retains the mid-saliency range while avoiding background noise and extreme outliers
    low_threshold = np.percentile(saliency_flat, 20)
    high_threshold = np.percentile(saliency_flat, 80)
    
    mask = (saliency_flat > low_threshold) & (saliency_flat < high_threshold)
    
    #Ensure enough samples remain; otherwise fallback to a higher percentile
    if np.sum(mask) < 250:  # 5 colors ** 50 samples minimum
        threshold = np.percentile(saliency_flat, 70)
        mask = saliency_flat > threshold
    
    pixels = pixels[mask]
    saliency_flat = saliency_flat[mask]

    # If still too few pixels remain, randomly sample from the full image
    if len(pixels) < 5:
        all_pixels = lab_image.reshape(-1, 3)
        all_saliency = saliency_map.flatten()
        indices = np.random.choice(len(all_pixels), size=min(1000, len(all_pixels)), replace=False)
        pixels = all_pixels[indices]
        saliency_flat = all_saliency[indices]

    # Improved initialization: select more diverse starting points
    # 1. First color: the most salient pixel
    idx_max_saliency = np.argmax(saliency_flat)
    p1 = pixels[idx_max_saliency]
    w1 = saliency_flat[idx_max_saliency]

    # 2. Second color: the pixel farthest from the first in Lab space
    distances = [weighted_lab_distance(p, p1, w1) for p in pixels]
    idx_farthest = np.argmax(distances)
    p2 = pixels[idx_farthest]
    w2 = saliency_flat[idx_farthest]

    selected = [(p1, w1), (p2, w2)]
    selected_indices = [idx_max_saliency, idx_farthest]

    # Prepare the remaining candidate pool
    mask_remaining = np.ones(len(pixels), dtype=bool)
    mask_remaining[selected_indices] = False
    candidates = pixels[mask_remaining]
    saliency_cand = saliency_flat[mask_remaining]

    # Iteratively pick the next colors based on maximal minimal distance
    # This encourages a diverse set of colors while accounting for saliency
    for _ in range(2, 5):
        max_min_dist = -1
        best_pixel = None
        best_weight = None
        best_idx = None
        
        for i, candidate in enumerate(candidates):
            # Compute the candidate’s minimal distance to the already selected colors
            min_dist = min([weighted_lab_distance(candidate, s[0], s[1]) for s in selected])
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_pixel = candidate
                best_weight = saliency_cand[i]
                best_idx = i
        
        if best_pixel is not None:
            selected.append((best_pixel, best_weight))
            # Remove the chosen candidate from the pool
            candidates = np.delete(candidates, best_idx, axis=0)
            saliency_cand = np.delete(saliency_cand, best_idx)

    # Convert selected colors from Lab back to RGB
    selected_lab = np.array([p for p, _ in selected], dtype=np.float32).reshape(-1, 1, 3)
    selected_rgb = lab_to_rgb(selected_lab)
    selected_rgb = selected_rgb.reshape(-1, 3)

    # Build the final palette list with RGB colors and their weights
    palette = [{"RGB": rgb.tolist(), "Weight": float(w)} 
               for rgb, (_, w) in zip(selected_rgb, selected)]
    
    return palette
