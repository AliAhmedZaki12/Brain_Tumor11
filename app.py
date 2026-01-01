# ===============================
# app.py  (PRODUCTION MEDICAL AI)
# ===============================

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "brain_tumor_model_lite.tfliteA"  
IMG_SIZE = (299, 299)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

CONF_THRESHOLD = 0.65        # حد أدنى للثقة
MARGIN_THRESHOLD = 0.15      # فرق الثقة بين أعلى فئتين
ENTROPY_THRESHOLD = 1.2      # عدم اليقين (OOD detection)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(MODEL_PATH)

# ===============================
# IMAGE PREPROCESS
# ===============================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# ENTROPY (UNCERTAINTY)
# ===============================
def entropy(probs):
    return -np.sum([p * math.log(p + 1e-8) for p in probs])

# ===============================
# PREDICTION (MEDICAL-GRADE)
# ===============================
def predict(image: Image.Image):
    img = preprocess_image(image)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    # ---- Metrics ----
    sorted_probs = np.sort(probs)[::-1]
    max_prob = sorted_probs[0]
    second_prob = sorted_probs[1]
    margin = max_prob - second_prob
    ent = entropy(probs)

    # ---- Medical Decision Logic ----
    is_uncertain = (
        max_prob < CONF_THRESHOLD or
        margin < MARGIN_THRESHOLD or
        ent > ENTROPY_THRESHOLD
    )

    if is_uncertain:
        final_probs = np.zeros_like(probs)
        final_probs[CLASS_NAMES.index("No Tumor")] = 1.0
        decision = "No Tumor (Low Confidence / OOD)"
    else:
        final_probs = probs
        decision = CLASS_NAMES[np.argmax(probs)]

    return final_probs, decision, max_prob, margin, ent

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title(" Brain Tumor Detection ")

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
        top_idx = np.argmax(probs)

        # ---- Result ----
        if "No Tumor" in decision:
            st.warning(f"🟡 {decision}")
        else:
            st.success(f" Tumor Type: {decision} ({max_prob*100:.2f}%)")

        # ---- Explainability ----
        st.markdown("### 🔍 Confidence Analysis")
        st.write(f"• Max Probability: **{max_prob:.2f}**")
        st.write(f"• Confidence Margin: **{margin:.2f}**")
        st.write(f"• Entropy (Uncertainty): **{ent:.2f}**")

        # ---- Table ----
        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        }).sort_values(by="Probability", ascending=False)

        st.table(df)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(df["Class"], df["Probability"])
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)
