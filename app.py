# =====================================================
# app.py  |  MEDICAL-GRADE BRAIN TUMOR DETECTION
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import cv2

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "brain_tumor_model_lite.tfliteA"   # ✅ صحيح
IMG_SIZE = (299, 299)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# ---- Safety thresholds ----
CONF_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.15
ENTROPY_THRESHOLD = 1.2

# =====================================================
# LOAD TFLITE MODEL
# =====================================================
@st.cache_resource
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model(MODEL_PATH)

# =====================================================
# IMAGE PREPROCESSING
# =====================================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# =====================================================
# IMAGE VALIDATION (MRI GATE)
# =====================================================
def is_valid_mri(image: Image.Image):
    gray = np.array(image.convert("L"))

    variance = np.var(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    intensity_range = gray.max() - gray.min()

    # ---- MRI Heuristics ----
    if variance > 3000:
        return False
    if edge_density > 0.15:
        return False
    if intensity_range > 220:
        return False

    return True

# =====================================================
# UNCERTAINTY (ENTROPY)
# =====================================================
def entropy(probs):
    return -np.sum([p * math.log(p + 1e-8) for p in probs])

# =====================================================
# MEDICAL-GRADE PREDICTION PIPELINE
# =====================================================
def predict(image: Image.Image):

    # ---------- STAGE 0: MRI VALIDATION ----------
    if not is_valid_mri(image):
        probs = np.zeros(len(CLASS_NAMES))
        probs[CLASS_NAMES.index("No Tumor")] = 1.0
        return probs, "No Tumor (Invalid / Non-MRI Image)", 0.0, 0.0, 0.0

    # ---------- STAGE 1: MODEL INFERENCE ----------
    img = preprocess_image(image)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    sorted_probs = np.sort(probs)[::-1]
    max_prob = sorted_probs[0]
    margin = sorted_probs[0] - sorted_probs[1]
    ent = entropy(probs)

    # ---------- STAGE 2: CONFIDENCE FILTER ----------
    uncertain = (
        max_prob < CONF_THRESHOLD or
        margin < MARGIN_THRESHOLD or
        ent > ENTROPY_THRESHOLD
    )

    if uncertain:
        final_probs = np.zeros_like(probs)
        final_probs[CLASS_NAMES.index("No Tumor")] = 1.0
        decision = "No Tumor (Low Confidence / OOD)"
    else:
        final_probs = probs
        decision = CLASS_NAMES[np.argmax(probs)]

    return final_probs, decision, max_prob, margin, ent

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection — Medical Grade AI")

uploaded_files = st.file_uploader(
    "Upload MRI Brain Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=file.name, use_column_width=True)

        probs, decision, max_prob, margin, ent = predict(image)

        # ---------- RESULT ----------
        if "No Tumor" in decision:
            st.warning(f"🟡 {decision}")
        else:
            st.success(f" Tumor Type: {decision} ({max_prob*100:.2f}%)")

        # ---------- CONFIDENCE ANALYSIS ----------
        st.markdown("### 🔍 Confidence Analysis")
        st.write(f"• Max Probability: **{max_prob:.2f}**")
        st.write(f"• Confidence Margin: **{margin:.2f}**")
        st.write(f"• Entropy (Uncertainty): **{ent:.2f}**")

        # ---------- TABLE ----------
        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        }).sort_values(by="Probability", ascending=False)

        st.table(df)

        # ---------- PLOT ----------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(df["Class"], df["Probability"])
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)
