# ============================================
# visualization/color_space_plot.py
# ============================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
from app.core.utils import rgb_to_lab

def plot_lab_space(palette):
    """
    Displays the extracted 5-color palette inside a 3D CIELab color space.
    """

    # Sanity check: the palette must contain exactly 5 entries
    if not palette or len(palette) != 5:
        st.error("❌ The palette must contain exactly 5 colors.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Convert palette RGB colors into Lab and validate their numerical ranges
    lab_colors = []
    lab_values = {'L': [], 'a': [], 'b': []}
    
    for i, color_info in enumerate(palette):
        try:
            rgb = np.array(color_info["RGB"], dtype=np.uint8).reshape(1, 1, 3)
            lab = rgb_to_lab(rgb)

            L_val = lab[0, 0, 0]
            a_val = lab[0, 0, 1]
            b_val = lab[0, 0, 2]

            # Check theoretical CIELab bounds and notify user when values fall outside
            if not (0 <= L_val <= 100):
                st.warning(f"⚠️ Color {i+1}: L*={L_val:.1f} is outside the expected range [0,100]")
            if not (-128 <= a_val <= 127):
                st.warning(f"⚠️ Color {i+1}: a*={a_val:.1f} is outside the expected range [-128,127]")
            if not (-128 <= b_val <= 127):
                st.warning(f"⚠️ Color {i+1}: b*={b_val:.1f} is outside the expected range [-128,127]")

            lab_colors.append({
                'L': L_val,
                'a': a_val,
                'b': b_val,
                'RGB': color_info["RGB"],
                'Weight': color_info["Weight"]
            })

            # Collect for axis scaling
            lab_values['L'].append(L_val)
            lab_values['a'].append(a_val)
            lab_values['b'].append(b_val)

        except Exception as e:
            st.error(f"❌ Lab conversion error for color {i+1}: {str(e)}")
            return

    # Compute axis limits dynamically based on palette data
    L_min, L_max = min(lab_values['L']), max(lab_values['L'])
    a_min, a_max = min(lab_values['a']), max(lab_values['a'])
    b_min, b_max = min(lab_values['b']), max(lab_values['b'])

    # Add a 10% margin to improve readability
    L_range = L_max - L_min
    a_range = a_max - a_min
    b_range = b_max - b_min

    L_min_plot = max(0, L_min - L_range * 0.1)
    L_max_plot = min(100, L_max + L_range * 0.1)
    a_min_plot = a_min - a_range * 0.1
    a_max_plot = a_max + a_range * 0.1
    b_min_plot = b_min - b_range * 0.1
    b_max_plot = b_max + b_range * 0.1

    # Plot each color as a 3D point in Lab space with size scaled by saliency weight
    max_weight = max(color['Weight'] for color in lab_colors)
    
    for i, color_lab in enumerate(lab_colors):
        rgb_normalized = np.array(color_lab['RGB']) / 255.0

        # Marker size scaled relative to the maximum saliency weight
        size = 50 + (color_lab['Weight'] / max_weight) * 200

        ax.scatter(
            color_lab['a'],        # a* axis (red ↔ green)
            color_lab['b'],        # b* axis (yellow ↔ blue)
            color_lab['L'],        # L* axis (lightness)
            c=[rgb_normalized],
            s=size,
            edgecolors='black',
            linewidth=1.5,
            alpha=0.9,
            label=f'Color {i+1} ({color_lab["Weight"]*100:.1f}%)'
        )

        # Optional helper line connecting the point to the mid-lightness plane (L*=50)
        ax.plot(
            [color_lab['a'], color_lab['a']],
            [color_lab['b'], color_lab['b']],
            [50, color_lab['L']],
            'k--', alpha=0.2, linewidth=0.8
        )

    # Apply limits and axis labels
    ax.set_xlabel("a* (red ↔ green)", fontsize=12, labelpad=10)
    ax.set_ylabel("b* (yellow ↔ blue)", fontsize=12, labelpad=10)
    ax.set_zlabel("L* (lightness)", fontsize=12, labelpad=10)

    ax.set_xlim(a_min_plot, a_max_plot)
    ax.set_ylim(b_min_plot, b_max_plot)
    ax.set_zlim(L_min_plot, L_max_plot)

    ax.set_title(
        "CIELab Space of Extracted Palette\n(Point size reflects relative saliency weight)",
        fontsize=14, pad=20
    )

    # Improve visual clarity
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)

    # Set a reasonably informative camera angle
    ax.view_init(elev=20, azim=45)

    # Explanation block for users
    st.markdown("""
    **Understanding the CIELab visualization:**

    - **L*** (vertical): Lightness → **0** = black, **50** = mid-grey, **100** = white  
    - **a*** (horizontal): Red/Green axis → **+a** = red, **-a** = green  
    - **b*** (depth): Yellow/Blue axis → **+b** = yellow, **-b** = blue  

    **Scale & Representation:**
    - **Point size** = proportional to relative saliency weight  
    - **Coordinates** = true CIELab values  
    - **Dashed lines** = projection toward mid-lightness (L* = 50)
    """)

    st.pyplot(fig)

    # Display detailed Lab values and palette diagnostics
    st.write("### Detailed CIELab Values:")

    st.write(f"**Value ranges:** L* [{L_min:.1f}-{L_max:.1f}], a* [{a_min:.1f}-{a_max:.1f}], b* [{b_min:.1f}-{b_max:.1f}]")

    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    for i, (color_lab, col) in enumerate(zip(lab_colors, columns)):
        with col:
            hex_color = "#{:02x}{:02x}{:02x}".format(*color_lab['RGB'])
            st.markdown(
                f'<div style="background-color:{hex_color};width:100%;height:20px;border-radius:5px;margin-bottom:5px;"></div>',
                unsafe_allow_html=True
            )

            st.markdown(f"**Color {i+1}**")
            st.markdown(f"L*: `{color_lab['L']:.1f}`")
            st.markdown(f"a*: `{color_lab['a']:.1f}`")
            st.markdown(f"b*: `{color_lab['b']:.1f}`")
            st.markdown(f"Weight: `{color_lab['Weight']*100:.1f}%`")

            # Flag whether values fall into standard CIELab constraints
            if (
                0 <= color_lab['L'] <= 100 and
                -128 <= color_lab['a'] <= 127 and
                -128 <= color_lab['b'] <= 127
            ):
                st.markdown("✅ Within expected range")
            else:
                st.markdown("⚠️ Outside expected range")

    # Additional distribution diagnostics
    st.write("### Distribution Diagnostics:")

    from scipy.spatial.distance import euclidean

    st.write("**Pairwise distances between colors in Lab space:**")
    distance_matrix = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            if i != j:
                dist = euclidean(
                    [lab_colors[i]['L'], lab_colors[i]['a'], lab_colors[i]['b']],
                    [lab_colors[j]['L'], lab_colors[j]['a'], lab_colors[j]['b']]
                )
                distance_matrix[i][j] = dist

    avg_distance = np.mean(distance_matrix[np.triu_indices(5, 1)])
    st.write(f"**Average color distance:** `{avg_distance:.1f}` Lab units")

    # Provide qualitative feedback on color diversity
    if avg_distance > 30:
        st.success("High chromatic diversity")
    elif avg_distance > 15:
        st.info("📊 Moderate chromatic diversity")
    else:
        st.warning("Low chromatic diversity — colors are very similar")
