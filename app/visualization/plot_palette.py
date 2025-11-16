# ============================================
# visualization/plot_palette.py
# ============================================
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from app.core.utils import rgb_to_hex

def show_palette(image, palette, title="Palette extraite"):
    """
    Display the extracted color palette next to the original image.
    """
    num_colors = len(palette)
    colors = np.array([p["RGB"] for p in palette], dtype=np.uint8)
    weights = np.array([p["Weight"] for p in palette])
    weights = weights / np.sum(weights)

    bar_height = 100
    bar_width = 500
    bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

    start = 0
    for i, (color, w) in enumerate(zip(colors, weights)):
        end = start + int(bar_width * w)
        bar[:, start:end, :] = color
        start = end

    # Build the visualization figure: original image + palette bar
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(bar)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    # Display HEX codes and weight percentages with unique Streamlit keys
    st.markdown("### 🎨 Couleurs extraites")
    cols = st.columns(num_colors)
    for i, (c, w) in enumerate(zip(colors, weights)):
        hex_code = rgb_to_hex(c)
        with cols[i]:
            # Ensures each color picker gets a unique Streamlit key
            st.color_picker(f"Couleur {i+1}", hex_code, disabled=True, key=f"picker_{i}")
            st.caption(f"{hex_code}\nPoids : {w*100:.1f}%")
