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
from tensorflow.keras.layers import Conv2D

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
    Upload any MRI image (any size or resolution).
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
# استخراج شكل الإدخال الحقيقي للنموذج (آمن 100%)
# =========================================================
def get_model_image_shape(model):
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    if len(shape) != 4:
        raise ValueError(f"Unsupported model input shape: {shape}")

    # channels_last
    if shape[-1] in (1, 3):
        return shape[1], shape[2], shape[3], "channels_last"

    # channels_first
    if shape[1] in (1, 3):
        return shape[2], shape[3], shape[1], "channels_first"

    raise ValueError(f"Cannot infer channels from shape: {shape}")

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CHANNEL_ORDER = get_model_image_shape(model)

st.caption(f"Detected model input shape: {model.input_shape}")

# =========================================================
# Preprocessing (أي صورة → حجم النموذج بدون تشويه)
# =========================================================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)

    # RGB / Grayscale
    if IMG_CHANNELS == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    original_size = image.size
    target_w, target_h = IMG_WIDTH, IMG_HEIGHT

    scale = min(target_w / image.width, target_h / image.height)
    new_w = int(image.width * scale)
    new_h = int(image.height * scale)

    resized = image.resize((new_w, new_h), Image.BILINEAR)

    padded = Image.new(image.mode, (target_w, target_h), 0)
    padded.paste(
        resized,
        ((target_w - new_w) // 2, (target_h - new_h) // 2)
    )

    img_array = np.array(padded, dtype=np.float32) / 255.0

    if IMG_CHANNELS == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    if CHANNEL_ORDER == "channels_first":
        img_array = np.transpose(img_array, (0, 3, 1, 2))

    return img_array, padded, original_size

# =========================================================
# تحديد آخر Conv Layer تلقائيًا (Grad-CAM Safe)
# =========================================================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model")

LAST_CONV_LAYER = find_last_conv_layer(model)
st.caption(f"Grad-CAM layer detected: {LAST_CONV_LAYER}")

# =========================================================
# Grad-CAM
# =========================================================
def make_gradcam_heatmap(img_array, model, conv_layer_name, pred_index):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

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
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img_array, display_image, original_size = preprocess_image(uploaded_file)

    st.subheader("📷 Uploaded Image")
    st.image(display_image, use_column_width=True)
    st.caption(f"Original size: {original_size[0]} × {original_size[1]}")

    # =====================================================
    # Prediction
    # =====================================================
    preds = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(preds))
    predicted_class = class_labels[predicted_index]
    confidence = preds[predicted_index] * 100

    st.subheader("🧠 Prediction Result")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # =====================================================
    # Probability Table
    # =====================================================
    st.subheader("📊 Class Probabilities")

    prob_df = pd.DataFrame({
        "Tumor Type": class_labels,
        "Probability (%)": preds * 100
    }).sort_values(by="Probability (%)", ascending=False)

    st.dataframe(
        prob_df.style.format({"Probability (%)": "{:.2f}"}),
        use_container_width=True
    )

    # =====================================================
    # Grad-CAM
    # =====================================================
    st.subheader("🔍 Model Attention (Grad-CAM)")

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        LAST_CONV_LAYER,
        predicted_index
    )

    gradcam_img = overlay_gradcam(display_image, heatmap)
    st.image(gradcam_img, use_column_width=True)

    # =====================================================
    # Disclaimer
    # =====================================================
    st.warning(
        "⚠️ This tool is for research and educational purposes only "
        "and must not be used as a medical diagnosis."
    )
