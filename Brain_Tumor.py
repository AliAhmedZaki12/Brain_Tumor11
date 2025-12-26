import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense

# ===============================
# ⚙️ App Configuration
# ===============================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ===============================
# 📦 Load Model (Cached)
# ===============================
@st.cache_resource
def load_model():
    # يجب أن يكون النموذج مبني على Xception مسبقًا
    return tf.keras.models.load_model("brain_tumor_model.h5")

# ===============================
# 📦 Load Class Labels
# ===============================
@st.cache_data
def load_class_labels():
    with open("class_labels.json", "r") as f:
        return json.load(f)

model = load_model()
class_labels = load_class_labels()

# ===============================
# 🧪 Image Preprocessing for Xception (299x299)
# ===============================
def preprocess_image(image: Image.Image, img_size=299):
    # تحويل الصورة لـ RGB وتغيير حجمها
    image = image.resize((img_size, img_size)).convert("RGB")
    
    # تحويل الصورة لمصفوفة numpy وتحويل dtype
    image = np.array(image, dtype=np.float32) / 255.0
    
    # إضافة batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

# ===============================
# 🎯 UI
# ===============================
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload an MRI image to detect the type of brain tumor.")

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed_image = preprocess_image(image)
            
            # التنبؤ
            predictions = model.predict(processed_image, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_labels[str(predicted_index)]
            confidence = predictions[predicted_index] * 100

        st.success(f"🧠 Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.markdown("### 📊 Class Probabilities")
        for i, prob in enumerate(predictions):
            st.write(f"{class_labels[str(i)]}: **{prob*100:.2f}%**")
            st.progress(float(prob))

