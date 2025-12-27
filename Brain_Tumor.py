# =====================================================
# 🧠 Brain Tumor Detection System
# 4-Class CNN (Softmax) – Final Streamlit App
# =====================================================

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pandas as pd

# =====================================================
# 🔧 Page Config
# =====================================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

# =====================================================
# 🔹 Load Model
# =====================================================
@st.cache_resource
def load_trained_model():
    return load_model("brain_tumor_4class.h5")

model = load_trained_model()

# =====================================================
# 🔹 Classes
# =====================================================
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# =====================================================
# 🔹 Image Preprocessing
# =====================================================
IMG_SIZE = 224

def preprocess_image(image: Image.Image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# =====================================================
# 🖥️ UI
# =====================================================
st.title("🧠 Brain Tumor Detection System")
st.write(
    "Upload an MRI image to get **real multi-class predictions** "
    "from a trained deep learning model."
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# 🔮 Prediction
# =====================================================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", width=350)

    processed = preprocess_image(image)

    # Softmax output
    preds = model.predict(processed, verbose=0)[0]

    # Normalize safety
    preds = preds / preds.sum()

    # =================================================
    # 📊 Results Table
    # =================================================
    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability (%)": np.round(preds * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities")
    st.dataframe(df, width=520)

    # Top Prediction
    top = df.iloc[0]

    if top["Tumor Type"] == "notumor":
        st.success(
            f"✅ **No Tumor Detected** "
            f"({top['Probability (%)']}% confidence)"
        )
    else:
        st.error(
            f"⚠️ **Tumor Detected: {top['Tumor Type']}** "
            f"({top['Probability (%)']}% confidence)"
        )

# =====================================================
# 🔻 Footer
# =====================================================
st.caption("Developed by Ali Ahmed Zaki")
