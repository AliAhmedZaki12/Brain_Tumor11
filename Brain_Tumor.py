# ===============================
# 🧠 Brain Tumor Detection App
# Binary model → 4-class adapter
# ===============================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pandas as pd

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# ===============================
# 🔹 Load Model
# ===============================
@st.cache_resource
def load_trained_model():
    return load_model("brain_tumor_model.h5", compile=False)

model = load_trained_model()

# ===============================
# 🔹 Fixed Classes
# ===============================
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ===============================
# 🔹 Detect Input
# ===============================
INPUT_SHAPE = model.input_shape

if len(INPUT_SHAPE) == 2:
    MODEL_TYPE = "VECTOR"
    VECTOR_SIZE = INPUT_SHAPE[1]
elif len(INPUT_SHAPE) == 4:
    MODEL_TYPE = "IMAGE"
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = INPUT_SHAPE[1:]
else:
    st.error(f"Unsupported model input shape: {INPUT_SHAPE}")
    st.stop()

# ===============================
# 🔹 Preprocess
# ===============================
def preprocess_image(image: Image.Image):
    image = np.array(image.convert("RGB"))

    if MODEL_TYPE == "IMAGE":
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    vec = image.flatten()

    if vec.shape[0] > VECTOR_SIZE:
        vec = vec[:VECTOR_SIZE]
    else:
        vec = np.pad(vec, (0, VECTOR_SIZE - vec.shape[0]))

    return np.expand_dims(vec, axis=0)

# ===============================
# 🔹 UI
# ===============================
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to get prediction probabilities")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=350)

    processed = preprocess_image(image)
    raw_pred = model.predict(processed, verbose=0)[0]

    # ===============================
    # 🔹 Binary → 4-Class Adapter
    # ===============================
    p_tumor = float(raw_pred[0])
    p_notumor = 1 - p_tumor

    tumor_share = p_tumor / 3

    preds = np.array([
        tumor_share,      # glioma
        tumor_share,      # meningioma
        p_notumor,        # notumor
        tumor_share       # pituitary
    ])

    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability (%)": np.round(preds * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities")
    st.dataframe(df, width=500)

    top = df.iloc[0]
    st.success(
        f"Model suggests **{top['Tumor Type']}** "
        f"with confidence **{top['Probability (%)']}%**"
    )

    st.warning(
        "⚠️ This model is binary (tumor / no tumor). "
        "Tumor subtype probabilities are approximated for visualization only."
    )

st.caption("Legacy binary model | Safe adapter | No retraining required")
