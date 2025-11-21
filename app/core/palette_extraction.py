# ============================================
# core/palette_extraction.py
# ============================================
import numpy as np
import cv2
from app.core.distance_metrics import weighted_lab_distance
from app.core.utils import rgb_to_lab, lab_to_rgb
from app.core.log_utils import get_logger

logger = get_logger(__name__)

def extract_palette(image, saliency_map):
    logger.info("Starting palette extraction")

    lab_image = rgb_to_lab(image)
    h, w, _ = lab_image.shape
    logger.debug(f"Image dimensions: {h}x{w}, total pixels: {h*w}")

    pixels = lab_image.reshape(-1, 3)
    saliency_flat = saliency_map.flatten()

    low_threshold = np.percentile(saliency_flat, 20)
    high_threshold = np.percentile(saliency_flat, 80)
    
    mask = (saliency_flat > low_threshold) & (saliency_flat < high_threshold)
    
    logger.debug(f"Saliency thresholds - low: {low_threshold:.3f}, high: {high_threshold:.3f}")
    logger.debug(f"Candidate pixels after filtering: {np.sum(mask)}/{len(mask)}")

    if np.sum(mask) < 250:
        threshold = np.percentile(saliency_flat, 70)
        mask = saliency_flat > threshold
        logger.warning(f"Low candidate count, using fallback threshold: {threshold:.3f}")

    pixels = pixels[mask]
    saliency_flat = saliency_flat[mask]

    if len(pixels) < 5:
        logger.warning("Very few candidate pixels, using random sampling")
        all_pixels = lab_image.reshape(-1, 3)
        all_saliency = saliency_map.flatten()
        indices = np.random.choice(len(all_pixels), size=min(1000, len(all_pixels)), replace=False)
        pixels = all_pixels[indices]
        saliency_flat = all_saliency[indices]

    idx_max_saliency = np.argmax(saliency_flat)
    p1 = pixels[idx_max_saliency]
    w1 = saliency_flat[idx_max_saliency]
    logger.debug(f"First color (most salient): weight={w1:.3f}")

    distances = [weighted_lab_distance(p, p1, w1) for p in pixels]
    idx_farthest = np.argmax(distances)
    p2 = pixels[idx_farthest]
    w2 = saliency_flat[idx_farthest]
    logger.debug(f"Second color (farthest): weight={w2:.3f}")

    selected = [(p1, w1), (p2, w2)]
    selected_indices = [idx_max_saliency, idx_farthest]

    mask_remaining = np.ones(len(pixels), dtype=bool)
    mask_remaining[selected_indices] = False
    candidates = pixels[mask_remaining]
    saliency_cand = saliency_flat[mask_remaining]

    for k in range(2, 5):
        max_min_dist = -1
        best_pixel = None
        best_weight = None
        best_idx = None
        
        for i, candidate in enumerate(candidates):
            min_dist = min([weighted_lab_distance(candidate, s[0], s[1]) for s in selected])
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_pixel = candidate
                best_weight = saliency_cand[i]
                best_idx = i
        
        if best_pixel is not None:
            selected.append((best_pixel, best_weight))
            candidates = np.delete(candidates, best_idx, axis=0)
            saliency_cand = np.delete(saliency_cand, best_idx)
            logger.debug(f"Color {k+1} selected: min_dist={max_min_dist:.3f}")

    selected_lab = np.array([p for p, _ in selected], dtype=np.float32).reshape(-1, 1, 3)
    selected_rgb = lab_to_rgb(selected_lab)
    selected_rgb = selected_rgb.reshape(-1, 3)

    palette = [{"RGB": rgb.tolist(), "Weight": float(w)} 
               for rgb, (_, w) in zip(selected_rgb, selected)]
    
    logger.info("✅ Palette extraction completed successfully")
    return palette