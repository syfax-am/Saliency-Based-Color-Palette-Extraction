# ============================================
# core/distance_metrics.py
# ============================================
import numpy as np

def weighted_lab_distance(p_i, p_j, w_pi):
    """
    Computes the weighted distance between two pixels in the CIELab space,
    following the formula (Eq. 3) from Jahanian et al. (2015).

    Args:
        p_i, p_j: np.array([L*, a*, b*])
        w_pi: saliency weight for pixel i

    Returns:
        distance (float)
    """
    # Using an approximated CIEDE2000-like behavior for perceptual consistency
    #instead of a simple Euclidean computation in Lab space
    delta_L = p_i[0] - p_j[0]
    delta_a = p_i[1] - p_j[1] 
    delta_b = p_i[2] - p_j[2]
    
    # Euclidean distance in Lab with basic perceptual weighting
    # a* and b* channels generally carry more perceptual color variation than L*
    distance = np.sqrt((delta_L * 0.5)**2 + delta_a**2 + delta_b**2)
    
    # Apply saliency-based weighting to emphasize more visually important pixels
    weighted_distance = w_pi * distance
    
    return weighted_distance
