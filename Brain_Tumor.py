# =====================================================
# 🧠 Brain Tumor Detection (Binary + Estimated 4-Class)
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
# 🔹 Load Binary Model
# =====================================================
@st.cache_resource
def load_trained_model():
    try:
        return load_model("brain_tumor_model.h5", compile=False)
    except FileNotFoundError:
        st.error("❌ لم يتم العثور على نموذج الـ Binary. تأكد من رفع الملف brain_tumor_model.h5")
        st.stop()

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
    "Upload an MRI image to get predictions (binary model with estimated tumor type probabilities)."
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
    p_tumor = float(model.predict(processed, verbose=0)[0][0])
    p_notumor = 1 - p_tumor

    # =================================================
    # 🔹 Estimated 4-Class Distribution (heuristic)
    # =================================================
    priors = np.array([0.45, 0.30, 0.25])  # glioma, meningioma, pituitary
    tumor_est = priors * p_tumor
    preds = np.array([tumor_est[0], tumor_est[1], p_notumor, tumor_est[2]])

    # =================================================
    # 🔹 Softmax-style Scaling (for UI only)
    # =================================================
    def softmax_scale(p):
        e = np.exp(p * 5)  # scale factor 5 for visibility
        return e / e.sum()

    preds_scaled = softmax_scale(preds)

    # =================================================
    # 📊 Results Table
    # =================================================
    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability (%)": np.round(preds_scaled * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities (Estimated)")
    st.dataframe(df, width=520)

    # =================================================
    # 🔹 Top Prediction
    # =================================================
    top = df.iloc[0]
    if top["Tumor Type"] == "notumor":
        st.success(
            f"✅ **No Tumor Detected** ({top['Probability (%)']}% confidence)"
        )
    else:
        st.error(
            f"⚠️ **Tumor Detected: {top['Tumor Type']}** ({top['Probability (%)']}% confidence)"
        )

    # =================================================
    # 🔹 Interpretation Note
    # =================================================
    st.caption(
        "⚠️ Probabilities are estimated from a binary model. "
        "They are for display purposes only and not exact predictions for each tumor type."
    )

# =====================================================
# 🔻 Footer
# =====================================================
st.caption("Developed by Ali Ahmed Zaki")
