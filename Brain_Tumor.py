# =========================================================
# 🧠 Brain Tumor MRI Classification - Streamlit App
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import json
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# =========================================================
# إعداد الصفحة
# =========================================================
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classification")
st.write(
    """
    Upload any MRI scan image (any size or resolution).
    The model will classify the tumor type and explain its decision using Grad-CAM.
    """
)

# =========================================================
# تحميل النموذج والفئات
# =========================================================
@st.cache_resource
def load_brain_tumor_model():
    return load_model("brain_tumor_model.h5", compile=False)

@st.cache_data
def load_class_labels():
    with open("class_labels.json", "r") as f:
        return json.load(f)

model = load_brain_tumor_model()
class_labels = load_class_labels()

# =========================================================
# استخراج حجم الإدخال الحقيقي للنموذج
# =========================================================
input_shape = model.input_shape  # (None, H, W, C)
IMG_HEIGHT = input_shape[1]
IMG_WIDTH  = input_shape[2]
IMG_CHANNELS = input_shape[3]

st.caption(f"Model input shape: {input_shape}")

# =========================================================
# معالجة الصور (Any size → Model size بدون تشويه)
# =========================================================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    original_w, original_h = image.size
    target_w, target_h = IMG_WIDTH, IMG_HEIGHT

    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized_image = image.resize((new_w, new_h), Image.BILINEAR)

    padded_image = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_image.paste(resized_image, (x_offset, y_offset))

    img_array = np.array(padded_image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, padded_image, image.size

# =========================================================
# Grad-CAM
# =========================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    image = np.array(image)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

# =========================================================
# رفع الصورة
# =========================================================
uploaded_file = st.file_uploader(
    "Upload an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    processed_image, display_image, original_size = preprocess_image(uploaded_file)

    st.subheader("📷 Uploaded Image")
    st.image(display_image, use_column_width=True)
    st.caption(f"Original size: {original_size[0]} × {original_size[1]}")

    # =====================================================
    # التنبؤ
    # =====================================================
    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_labels[predicted_index]
    confidence = predictions[predicted_index] * 100

    st.subheader("🧠 Prediction Result")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # =====================================================
    # جدول الاحتمالات
    # =====================================================
    st.subheader("📊 Class Probabilities")

    prob_df = pd.DataFrame({
        "Tumor Type": class_labels,
        "Probability (%)": predictions * 100
    }).sort_values(by="Probability (%)", ascending=False)

    st.dataframe(
        prob_df.style.format({"Probability (%)": "{:.2f}"}),
        use_container_width=True
    )

    # =====================================================
    # Grad-CAM Visualization
    # =====================================================
    st.subheader("🔍 Model Attention (Grad-CAM)")

    # ⚠️ عدّل الاسم لو نموذجك ليس Xception
    last_conv_layer_name = "block14_sepconv2_act"

    heatmap = make_gradcam_heatmap(
        processed_image,
        model,
        last_conv_layer_name,
        predicted_index
    )

    gradcam_image = overlay_gradcam(display_image, heatmap)
    st.image(gradcam_image, caption="Grad-CAM Visualization", use_column_width=True)

    # =====================================================
    # Disclaimer
    # =====================================================
    st.warning(
        "⚠️ This tool is for research and educational purposes only. "
        "It is NOT a substitute for professional medical diagnosis."
    )
