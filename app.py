import streamlit as st
from PIL import Image
import numpy as np
import cv2
from utils import detect_faces, blur_faces
import tempfile

st.set_page_config(page_title="Face Blur App", layout="centered")
st.title("🕵️ Face Blur App (Image Mode)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original Image", use_column_width=True)

    # Controls
    st.sidebar.title("⚙️ Settings")
    style = st.sidebar.selectbox("Masking Style", ["blur", "pixelate", "blackbar"])
    blur_strength = st.sidebar.slider("Blur Intensity (for Gaussian)", 15, 99, 51, step=2)
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=False)

    # Process
    faces = detect_faces(img_bgr)
    result = blur_faces(img_bgr, faces, style=style, intensity=blur_strength, show_box=show_boxes)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    st.subheader("🔍 Result")
    st.image(result_pil, use_column_width=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        result_pil.save(tmp.name)
        st.download_button("⬇️ Download Blurred Image", open(tmp.name, "rb"), file_name="blurred.png", mime="image/png")
