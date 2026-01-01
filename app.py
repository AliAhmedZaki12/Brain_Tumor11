import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 🔹 Config
# ===============================
TFLITE_MODEL_PATH = "brain_tumor_model_lite.tfliteA"
IMG_SIZE = (299, 299)
CLASS_LABELS = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]  # عدّل حسب ترتيبك في التدريب

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ===============================
# 🔹 Load TFLite Model
# ===============================
@st.cache_resource
def load_tflite_model(model_url):
    import requests, io
    # تحميل النموذج من GitHub
    response = requests.get(model_url)
    tflite_model = response.content
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(TFLITE_MODEL_PATH)
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===============================
# 🔹 Helper Functions
# ===============================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)
    return img_array

def predict_tflite(image_array: np.ndarray):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

def is_brain_mri(image: Image.Image, gray_ratio_threshold=0.7):
    """
    Rejects images that are not brain MRIs based on low saturation (grayness)
    """
    image = image.convert("RGB").resize(IMG_SIZE)
    img_np = np.array(image)/255.0
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = max_rgb - min_rgb
    gray_ratio = np.sum(saturation < 0.2) / (IMG_SIZE[0]*IMG_SIZE[1])
    return gray_ratio >= gray_ratio_threshold

# ===============================
# 🔹 Streamlit App
# ===============================
st.title("🧠 Brain Tumor Classification (TFLite)")
st.write("Upload a brain MRI image. Non-brain images are automatically rejected.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---- Reject non-brain images ----
    if not is_brain_mri(image):
        st.error("❌ Rejected: Not a valid brain MRI image.")
    else:
        st.success("✅ Image accepted, running prediction...")

        # ---- Preprocess & Predict ----
        img_array = preprocess_image(image)
        probs = predict_tflite(img_array)
        top_idx = np.argmax(probs)
        top_label = CLASS_LABELS[top_idx]

        # ---- Display Probabilities Table ----
        df_probs = pd.DataFrame({
            "Class": CLASS_LABELS,
            "Probability": np.round(probs*100, 2)
        }).sort_values("Probability", ascending=False)
        st.subheader("Prediction Probabilities")
        st.dataframe(df_probs)

        # ---- Highlight Top Class ----
        st.markdown(f"### 🔹 Predicted Tumor Type: **{top_label}** ({probs[top_idx]*100:.2f}%)")

        # ---- Plot Bar Chart ----
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(df_probs['Class'], df_probs['Probability'], color='skyblue')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        for i, v in enumerate(df_probs['Probability']):
            ax.text(i, v+1, f"{v:.1f}%", ha='center', fontweight='bold')
        st.pyplot(fig)
