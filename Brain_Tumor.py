# ===============================
# 🧠 Brain Tumor Detection App
# ===============================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pandas as pd

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

# ===============================
# 🔹 Load Model
# ===============================
@st.cache_resource
def load_trained_model():
    return load_model("brain_tumor_model.h5", compile=False)

model = load_trained_model()

# ===============================
# 🔹 Robust Input Shape Extraction
# ===============================
def get_model_input_details(model):
    """
    Safely extract image shape and channel order
    """
    shape = model.inputs[0].shape  # TensorShape

    # Convert TensorShape → tuple
    shape = tuple(dim if dim is not None else -1 for dim in shape)

    if len(shape) != 4:
        raise ValueError(f"Model input must be 4D, got: {shape}")

    # channels_last → (None, H, W, C)
    if shape[-1] in [1, 3]:
        return shape[1], shape[2], shape[3], "channels_last"

    # channels_first → (None, C, H, W)
    if shape[1] in [1, 3]:
        return shape[2], shape[3], shape[1], "channels_first"

    raise ValueError(f"Unsupported input shape: {shape}")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CHANNEL_ORDER = get_model_input_details(model)

# ===============================
# 🔹 Image Preprocessing
# ===============================
def preprocess_image(image: Image.Image):
    image = np.array(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w, _ = image.shape
    scale = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    y_offset = (IMG_HEIGHT - new_h) // 2
    x_offset = (IMG_WIDTH - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    canvas = canvas.astype("float32") / 255.0

    if IMG_CHANNELS == 1:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        canvas = np.expand_dims(canvas, axis=-1)

    if CHANNEL_ORDER == "channels_first":
        canvas = np.transpose(canvas, (2, 0, 1))

    return np.expand_dims(canvas, axis=0)

# ===============================
# 🔹 UI
# ===============================
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to get prediction probabilities")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess_image(image)

    preds = model.predict(processed, verbose=0)[0]

    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability (%)": np.round(preds * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities")
    st.dataframe(df, use_container_width=True)

    st.subheader("🩺 Medical Interpretation")
    top_class = df.iloc[0]["Tumor Type"]
    confidence = df.iloc[0]["Probability (%)"]

    st.success(
        f"Model suggests **{top_class}** with confidence **{confidence}%**"
    )

st.caption("Academic & Production Ready | CNN-based MRI Analysis")
