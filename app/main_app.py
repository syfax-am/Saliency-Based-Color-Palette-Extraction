# ============================================
# app/main_app.py ✅ FINAL VERSION - PERFECT IMAGE CENTERING
# ============================================
import sys
import pathlib
import io
import base64

# --- Automatically add the project root to PYTHONPATH ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Internal imports
from app.core.saliency import compute_combined_saliency
from app.core.palette_extraction import extract_palette
from app.visualization.plot_palette import show_palette
from app.visualization.color_space_plot import plot_lab_space
from app.core.log_utils import get_logger

logger = get_logger(__name__)

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def rgb_to_cmyk(rgb):
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    
    k = 1 - max(r, g, b)
    if k == 1:
        return 0, 0, 0, 100
    
    c = (1 - r - k) / (1 - k) * 100
    m = (1 - g - k) / (1 - k) * 100
    y = (1 - b - k) / (1 - k) * 100
    k = k * 100
    
    return int(c), int(m), int(y), int(k)

st.set_page_config(
    page_title="Saliency-Based Palette Extraction",
    page_icon=str(ROOT / "src" / "logo.ico"),
    layout="wide"
)

APP_VERSION = "v1.0.0"

logo_path = ROOT / "src" / "logo.png"

try:
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_bytes = f.read()
        logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")
        logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" width="38" style="margin-right: 10px; vertical-align: middle;">'
        
        st.markdown(
            f'''
            <div style="display: flex; align-items: center;">
                {logo_img_tag}
                <h1 style="display: inline; margin: 0;">Saliency-Based Palette Extraction</h1>
                <span style="margin-left: 10px; font-size: 0.9em; color: #666;">{APP_VERSION}</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
        logger.info("✅ Logo loaded successfully")
except Exception as e:
    st.title("Saliency-Based Palette Extraction")
    st.error(f"Could not load logo: {e}")
    logger.error(f"❌ Failed to load logo: {str(e)}")

st.caption("Implementation inspired by *Jahanian et al., 2015 — Purdue University*")

description_html = f"""
<div style="max-width: 100%; line-height: 1.6; font-size: 1rem;">
    This application extracts a <b>perceptually meaningful color palette</b> from an image
    by analyzing its <b>visual saliency map</b> to separate foreground and background.<br>
    The selected colors correspond to those that <b>draw the most human attention</b>,
    as described in the paper:
    <blockquote>
        Ali Jahanian, S.V.N. Vishwanathan, and J.P. Allebach, 
        "Autonomous Color Theme Extraction From Images Using Saliency", IS&T Electronic Imaging, 2015.
    </blockquote>
</div>
"""

st.markdown(description_html, unsafe_allow_html=True)

pdf_path = ROOT / "src" / "Autonomous Color Theme Extraction From Images Using Saliency.pdf"
if pdf_path.exists():
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    st.download_button(
        label="Download PDF Article",
        data=pdf_bytes,
        file_name="Autonomous_Color_Theme_Extraction.pdf",
        mime="application/pdf"
    )
    logger.info("✅ PDF article found and download button added")
else:
    st.warning("PDF article not found in src/")
    logger.warning("⚠️ PDF article not found in src/")

uploaded_file = st.file_uploader("📁 Choose an image (jpg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    logger.info(f"Image uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")
    st.write(f"**Uploaded file:** {uploaded_file.name}")

    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        logger.info(f"Image loaded successfully: {image_np.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load image: {str(e)}")
        st.error(f"❌ Error loading image: {str(e)}")
        st.stop()

    with st.spinner("Computing saliency map and extracting colors..."):
        logger.info("Starting saliency computation")
        saliency_map = compute_combined_saliency(image_np)
        logger.info(f"✅ Saliency computed - min: {saliency_map.min():.3f}, max: {saliency_map.max():.3f}, mean: {saliency_map.mean():.3f}")
        
        try:
            palette = extract_palette(image_np, saliency_map)
            logger.info(f"Palette extracted with {len(palette)} colors")
        except Exception as e:
            logger.error(f"❌ Error during extraction: {str(e)}", exc_info=True)
            st.error(f"❌ Error during extraction: {str(e)}")
            st.stop()

    total_weight = sum(color_info['Weight'] for color_info in palette)
    normalized_palette = []
    for color_info in palette:
        normalized_color = color_info.copy()
        normalized_color['Weight'] = color_info['Weight'] / total_weight
        normalized_palette.append(normalized_color)
    
    palette = normalized_palette
    logger.info(f"Palette weights normalized: {[f'{c['Weight']*100:.1f}%' for c in palette]}")

    st.write("## Extracted Colors")
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col3:
        st.image(image_np, caption="Original Image", width=300)
    
    cols = st.columns(5)
    
    for i, color_info in enumerate(palette):
        rgb = np.array(color_info["RGB"], dtype=int)
        hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
        cmyk = rgb_to_cmyk(rgb)
        
        with cols[i]:
            st.markdown(f"**{color_info['Weight']*100:.1f}%**")
            
            st.markdown(
                f'<div style="width: 100%; height: 80px; background-color: {hex_color}; '
                f'border-radius: 10px; border: 2px solid #ddd; margin-bottom: 10px; cursor: pointer;" '
                f'title="Click to copy {hex_color}"></div>',
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <style>
                div[data-testid="stTextInput"] input:disabled {{
                    cursor: text !important;
                    color: white !important;
                    -webkit-text-fill-color: white !important;
                    opacity: 1 !important;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.text_input("HEX", hex_color, key=f"hex_{i}", disabled=True)
            st.text_input("RGB", f"{rgb[0]}, {rgb[1]}, {rgb[2]}", key=f"rgb_{i}", disabled=True)
            st.text_input("CMYK", f"{cmyk[0]}%, {cmyk[1]}%, {cmyk[2]}%, {cmyk[3]}%", key=f"cmyk_{i}", disabled=True)

    st.write("")  
    
    bar_width = 800
    bar_height = 60
    proportion_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
    
    current_position = 0
    color_positions = []
    
    for i, color_info in enumerate(palette):
        color_positions.append({
            'position': current_position,
            'color': np.array(color_info["RGB"], dtype=float)
        })
        current_position += color_info["Weight"]
    
    color_positions.append({
        'position': 1.0,
        'color': np.array(palette[-1]["RGB"], dtype=float)
    })
    
    for x in range(bar_width):
        pos = x / (bar_width - 1)
        
        for i in range(len(color_positions) - 1):
            if color_positions[i]['position'] <= pos <= color_positions[i + 1]['position']:
                segment_start = color_positions[i]['position']
                segment_end = color_positions[i + 1]['position']
                segment_length = segment_end - segment_start
                
                if segment_length > 0:
                    ratio = (pos - segment_start) / segment_length
                    start_color = color_positions[i]['color']
                    end_color = color_positions[i + 1]['color']
                    interpolated_color = (1 - ratio) * start_color + ratio * end_color
                    proportion_bar[:, x] = np.clip(interpolated_color, 0, 255).astype(np.uint8)
                else:
                    proportion_bar[:, x] = color_positions[i]['color'].astype(np.uint8)
                break
    
    proportion_image = Image.fromarray(proportion_bar)
    
    bar_with_marks = Image.new('RGB', (bar_width, bar_height + 30), color=(255, 255, 255))
    bar_with_marks.paste(proportion_image, (0, 0))
    
    draw = ImageDraw.Draw(bar_with_marks)
    
    marks = [0.25, 0.50, 0.75]
    mark_positions = [int(bar_width * mark) for mark in marks]
    
    mark_color = (255, 255, 255)
    mark_width = 2
    
    for i, pos in enumerate(mark_positions):
        draw.line([(pos, 0), (pos, bar_height)], fill=mark_color, width=mark_width)
        
        percentage_text = f"{marks[i]*100:.0f}%"
        text_width = draw.textlength(percentage_text)
        text_position = (pos - text_width/2, bar_height + 5)
        draw.text(text_position, percentage_text, fill=(100, 100, 100))
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <div style="width: {bar_width}px; border-radius: 15px; overflow: hidden; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <img src="data:image/png;base64,{image_to_base64(bar_with_marks)}" 
                     style="width: 100%; height: auto;" />
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<p style="text-align: center; color: #666; margin-top: -10px;">'
        'Color distribution according to visual importance</p>',
        unsafe_allow_html=True
    )

    st.write("---")
    st.write(f"**Saliency Statistics** - Min: {saliency_map.min():.3f}, Max: {saliency_map.max():.3f}, Mean: {saliency_map.mean():.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_np, caption="Original Image", use_container_width=True)
    
    with col2:
        saliency_display = cv2.applyColorMap((saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        st.image(saliency_display, caption="Combined Saliency Map (JET colormap)", use_container_width=True)

    with st.expander("🧭 CIELab Space Visualization (scientific)", expanded=False):
        logger.info("Generating CIELab space visualization")
        plot_lab_space(palette)

else:
    logger.info("ℹ️ No image uploaded yet - waiting for user input")
    st.info("Please upload an image to begin.")

st.markdown("""
<script>
function setupColorCopy() {
    const colorDivs = document.querySelectorAll('div[style*="background-color"]');
    colorDivs.forEach(div => {
        div.addEventListener('click', function() {
            const style = this.getAttribute('style');
            const hexMatch = style.match(/background-color: (#[0-9a-fA-F]{6});/);
            if (hexMatch) {
                const hexColor = hexMatch[1];
                navigator.clipboard.writeText(hexColor).then(() => {
                    const originalTitle = this.getAttribute('title') || '';
                    this.setAttribute('title', 'Copied!');
                    setTimeout(() => {
                        this.setAttribute('title', originalTitle);
                    }, 2000);
                });
            }
        });
        div.setAttribute('title', 'Click to copy the color code');
        div.style.cursor = 'pointer';
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupColorCopy);
} else {
    setupColorCopy();
}

const observer = new MutationObserver(setupColorCopy);
observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

st.markdown("---")

email_icon = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
  <path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V4Zm2-1a1 1 0 0 0-1 1v.217l7 4.2 7-4.2V4a1 1 0 0 0-1-1H2Zm13 2.383-4.708 2.825L15 11.105V5.383Zm-.034 6.876-5.64-3.471L8 9.583l-1.326-.795-5.64 3.47A1 1 0 0 0 2 13h12a1 1 0 0 0 .966-.741ZM1 11.105l4.708-2.897L1 5.383v5.722Z"/>
</svg>
"""

linkedin_icon = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
  <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
</svg>
"""

github_icon = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
</svg>
"""

portfolio_icon = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
  <path d="M6.5 1A1.5 1.5 0 0 0 5 2.5V3H1.5A1.5 1.5 0 0 0 0 4.5v8A1.5 1.5 0 0 0 1.5 14h13a1.5 1.5 0 0 0 1.5-1.5v-8A1.5 1.5 0 0 0 14.5 3H11v-.5A1.5 1.5 0 0 0 9.5 1h-3zm0 1h3a.5.5 0 0 1 .5.5V3H6v-.5a.5.5 0 0 1 .5-.5zm1.886 6.914L15 7.151V12.5a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5V7.15l6.614 1.764a1.5 1.5 0 0 0 .772 0zM1.5 4h13a.5.5 0 0 1 .5.5v1.616L8.129 6.948a.5.5 0 0 1-.258 0L1 6.116V4.5a.5.5 0 0 1 .5-.5z"/>
</svg>
"""

with st.expander("**🟢 Interested in my skills? Let's work together!**", expanded=False):
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center;">
          <a href="mailto:syfaxaitmedjber@gmail.com?subject=Contact%20from%20Palette%20App&body=Hello%20Syfax,%0A%0AI%20am%20interested%20in%20your%20skills...">
            {email_icon}<br/>
            <strong>Email</strong>
          </a>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
          <a href="https://www.linkedin.com/in/syfax-ait-medjber/" target="_blank">
            {linkedin_icon}<br/>
            <strong>LinkedIn</strong>
          </a>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div style="text-align: center;">
          <a href="https://github.com/syfax-am/" target="_blank">
            {github_icon}<br/>
            <strong>GitHub</strong>
          </a>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div style="text-align: center;">
          <a href="https://my-portfolio.com" target="_blank">
            {portfolio_icon}<br/>
            <strong>Portfolio</strong>
          </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "*I’m passionate about Data Science and AI - especially NLP, Computer Vision, and model deployment. Always happy to connect, share ideas, or collaborate on meaningful projects.*"
    )

logger.info("Streamlit app startup completed")