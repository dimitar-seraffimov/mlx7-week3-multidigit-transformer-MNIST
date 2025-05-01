import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from s03_inference import DigitPredictor
import time

st.set_page_config(page_title="🧮 MultiDigit Canvas App", layout="wide")
st.title("✍️ Draw Multi-Digit Input")

predictor = DigitPredictor("multidigit_transformer.pth")

st.markdown("Draw a 56x56 grayscale digit combination in the canvas below and press **Predict**.")

# --- Clear Canvas Button ---
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = f"canvas_{int(time.time())}"

if st.button("🗑️ Clear Canvas"):
    st.session_state.canvas_key = f"canvas_{int(time.time())}"
    st.rerun()

# --- Canvas Drawing ---
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=8,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key,
    update_streamlit=True
)

if canvas_result.image_data is not None:
    image_data = canvas_result.image_data.astype(np.uint8)
    image = Image.fromarray(image_data).convert("L")
    resized = image.resize((56, 56))

    st.image(resized, caption="56x56 Resized Image", width=140)

    tensor = transforms.ToTensor()(resized).squeeze(0)
    patches = tensor.unfold(0, 14, 14).unfold(1, 14, 14)
    patches = patches.contiguous().view(-1, 14 * 14)
    input_tensor = patches.unsqueeze(0)  # (1, 16, 196)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            pred_tokens = predictor.predict(input_tensor)
        st.success("Prediction complete")
        st.markdown(f"### 🔢 Predicted Sequence: {' '.join(map(str, pred_tokens))}")
else:
    st.info("Draw something in the canvas to enable prediction.")
