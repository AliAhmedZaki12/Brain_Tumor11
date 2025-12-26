# app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

# ==============================
# إعداد الصفحة
# ==============================
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    layout="centered"
)
st.title("🧠 Brain Tumor MRI Classification")
st.write("""
Upload any MRI scan image and the model will predict the tumor type.
The app automatically resizes the image to 299x299 for the Xception model.
""")

# ==============================
# تحميل النموذج وملف الفئات
# ==============================
@st.cache_resource
def load_brain_tumor_model():
    model = load_model("brain_tumor_model.h5")
    return model

@st.cache_data
def load_class_labels():
    with open("class_labels.json", "r") as f:
        return json.load(f)

model = load_brain_tumor_model()
class_labels = load_class_labels()

# ==============================
# دالة معالجة الصورة
# ==============================
def preprocess_image(uploaded_file, target_size=(299, 299)):
    """
    تحول الصورة لأي حجم إلى RGB، تعيد تحجيمها، وتضيف بعد batch.
    """
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # تطبيع
    image_array = np.expand_dims(image_array, axis=0)  # batch dimension
    return image_array, image

# ==============================
# رفع الصورة والتنبؤ
# ==============================
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # معالجة الصورة
    processed_image, display_image = preprocess_image(uploaded_file)
    
    # عرض الصورة الأصلية بعد التحجيم
    st.image(display_image, caption="Uploaded MRI Image", use_column_width=True)
    
    # تنبؤ النموذج
    predictions = model.predict(processed_image, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    
    # عرض النتيجة
    st.subheader("Prediction Result")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    # عرض شريط ثقة لكل فئة
    st.subheader("Confidence for All Classes")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {predictions[0][i]*100:.2f}%")
