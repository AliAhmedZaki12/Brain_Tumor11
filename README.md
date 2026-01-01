# 🧠 Brain Tumor MRI Classification 

App : (https://braintumor11-fpmmhttkal22ckssmu7tnv.streamlit.app/)

## 📌 Project Overview

This project presents a **deep learning–based medical imaging system** for classifying **brain MRI images** into four clinically relevant categories.
The model leverages **transfer learning with the Xception architecture**, pretrained on ImageNet, and is optimized for **deployment on Streamlit Cloud** using a **lightweight TensorFlow Lite (TFLite) model**.

The system is designed with **practical deployment constraints in mind**, including:

* Reduced model size
* Fast inference
* Robust handling of non-brain images (classified safely as *No Tumor*)

⚠️ **Important Medical Disclaimer**
This application is intended for **research and educational purposes only** and **must not** be used as a diagnostic tool.

---

## 📌 Dataset

Brain MRI dataset structured as:

```
Training/
 ├── Glioma
 ├── Meningioma
 ├── Pituitary
 └── No Tumor

Testing/
 ├── Glioma
 ├── Meningioma
 ├── Pituitary
 └── No Tumor
```

### Data Splitting

* **Training set**
* **Validation set**
* **Test set**
* Stratified splitting ensures balanced class distribution.

---

## 📌 Model Architecture

### Base Network

* **Xception** (pretrained on ImageNet)
* `include_top=False`
* `pooling="max"`

### Custom Classifier Head

```
Flatten
Dropout(0.3)
Dense(128, ReLU)
Dropout(0.25)
Dense(4, Softmax)
```

### Training Configuration

* Image size: **299 × 299**
* Batch size: **32**
* Epochs: **10**
* Optimizer: **Adamax**
* Learning rate: **0.001**
* Loss: **Categorical Crossentropy**

### Metrics

* Accuracy
* Precision
* Recall

---

## 🔧 Image Preprocessing

* Pixel rescaling to **[0, 1]**
* Brightness augmentation (training only)
* No augmentation for validation/testing

---

## 📊 Evaluation

The model is evaluated using:

* **Confusion Matrix**
* **Classification Report**
* **Per-class Precision, Recall, F1-score**

### Example Performance

```
Train Accuracy:      ~94.5%
Validation Accuracy:~91.6%
Test Accuracy:      ~91.0%
```

---

## 🚀 Deployment (Streamlit + TFLite)

### Why TensorFlow Lite?

* Smaller model size (GitHub-friendly)
* Faster inference
* Fully compatible with **Streamlit Cloud**

The trained model is converted to:

```
brain_tumor_model_lite.tfliteA
```

> The `.tfliteA` extension is preserved intentionally and handled explicitly in code.

---

## 🖥 Streamlit Application Features

* Upload **single or multiple images**
* Automatic preprocessing
* **Per-image prediction**
* Output includes:

  * Final predicted class
  * Confidence score
  * Probability table for all classes
  * Horizontal bar chart visualization
* **Robust safety behavior**:

  * Images that do not resemble brain MRI patterns are automatically classified as **No Tumor**
  * Prevents misleading high-confidence predictions on irrelevant images

---

## 📦 Installation

```bash
pip install streamlit tensorflow numpy pandas pillow matplotlib opencv-python-headless
```

---

## ▶️ Usage

### Local

```bash
streamlit run app.py
```

### Streamlit Cloud

* Push the following files to GitHub:

  * `app.py`
  * `requirements.txt`
  * `brain_tumor_model_lite.tfliteA`
* Connect the repository to Streamlit Cloud
* Deploy 🚀

---

## 📁 Repository Structure

```
.
├── app.py
├── requirements.txt
├── brain_tumor_model_lite.tfliteA
└── README.md
```

---

## 🔬 Key Design Decisions

* ❌ No artificial confidence amplification
* ❌ No unsafe biasing of probabilities
* ✅ Real softmax outputs only
* ✅ Conservative handling of unknown / irrelevant images
* ✅ Medical-safety–oriented behavior

---

## 🧪 Limitations

* Model is trained on a specific MRI dataset
* Generalization to other imaging protocols is not guaranteed
* Not suitable for clinical diagnosis

---
