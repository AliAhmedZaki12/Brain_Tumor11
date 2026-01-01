# ===============================
# app.py
# ===============================
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# ===============================
# Constants
# ===============================
MODEL_PATH = "brain_tumor_model_lite.tfliteA"
IMG_SIZE = (299, 299)
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CONFIDENCE_THRESHOLD = 0.6  # أقل من هذا يتم تحويل الصورة تلقائيًا لـ "No Tumor"

# ===============================
# Load TFLite Model
# ===============================
@st.cache_resource
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(MODEL_PATH)

# ===============================
# Prediction Utilities
# ===============================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image: Image.Image, threshold=CONFIDENCE_THRESHOLD):
    """Predict probabilities for each class. Automatically assign No Tumor
       if the model is not confident."""
    img_array = preprocess_image(image)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    # ✅ أي صورة لا تتجاوز الاحتمال الأعلى threshold تُعامل كـ No Tumor
    max_prob = np.max(preds)
    if max_prob < threshold:
        preds = np.zeros_like(preds)
        preds[CLASS_NAMES.index("No Tumor")] = 1.0

    return preds

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection")

uploaded_files = st.file_uploader(
    "Upload MRI Brain Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        
        preds = predict(image)
        top_idx = np.argmax(preds)
        top_conf = preds[top_idx]

        st.success(f"Prediction: {CLASS_NAMES[top_idx]} ({top_conf*100:.2f}%)")
        
        # ✅ Table of probabilities
        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": preds
        }).sort_values(by="Probability", ascending=False)
        st.table(prob_df)
        
        # ✅ Bar chart
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(prob_df["Class"], prob_df["Probability"], color='skyblue')
        ax.set_xlim(0, 1)
        for i, v in enumerate(prob_df["Probability"]):
            ax.text(v + 0.01, i, f"{v*100:.1f}%", color='blue', fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)
