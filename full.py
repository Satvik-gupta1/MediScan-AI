import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorflow as tf
import torchxrayvision as xrv


# ---------------------------------------------------------------
# 1. Organ Classifier (MATCHES YOUR TRAINING MODEL EXACTLY)
# ---------------------------------------------------------------
class OrganClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        base = xrv.models.DenseNet(weights="densenet121-res224-chex")
        self.backbone = base.features                     # MATCHES .pth keys
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------
# 2. Load All 3 Models
# ---------------------------------------------------------------
@st.cache_resource
def load_models():

    # Organ classifier
    organ_model = OrganClassifier(num_classes=4)
    organ_model.load_state_dict(torch.load("densenet121_xray_organs.pth", map_location="cpu"))
    organ_model.eval()

    # Pneumonia model → expects (128,128,3)
    pneumonia_model = tf.keras.models.load_model("pneumonia_model.h5", compile=False)

    # Bone fracture model → softmax 2 classes
    fracture_model = tf.keras.models.load_model("bone_fracture_model.h5", compile=False)

    return organ_model, pneumonia_model, fracture_model


organ_model, pneumonia_model, fracture_model = load_models()


# ---------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------
torch_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_torch(img):
    return torch_transform(img).unsqueeze(0)


# ---------- Pneumonia preprocessing (128x128x3) ----------
def preprocess_pneumonia(img):
    img = img.resize((128, 128))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------- Bone fracture preprocessing ----------
def preprocess_fracture(img):
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------------------------------------------------------------
# 4. Prediction functions
# ---------------------------------------------------------------

# Organ detection
def predict_organ(img):
    with torch.no_grad():
        out = organ_model(img)
        prob = torch.softmax(out, dim=1)
        idx = torch.argmax(prob).item()
        conf = prob[0][idx].item()

    classes = ["bone", "brain", "heart", "lungs"]
    return classes[idx], conf


# ---------- Pneumonia prediction (SIGMOID MODEL) ----------
def predict_pneumonia(img):
    pred = pneumonia_model.predict(img)[0][0]  # sigmoid output

    if pred > 0.5:
        return "Pneumonia", pred
    else:
        return "Normal", 1 - pred


# Bone fracture detection (softmax: [no fracture, fracture])
def predict_fracture(img):
    pred = fracture_model.predict(img)[0]
    label = "Fracture" if pred[1] > pred[0] else "No Fracture"
    conf = float(max(pred))
    return label, conf


# ---------------------------------------------------------------
# 5. Streamlit UI
# ---------------------------------------------------------------
st.title("Unified Medical Imaging Diagnosis System 🏥")
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # STEP 1: Organ detection
    img_torch = preprocess_torch(image)
    organ, organ_conf = predict_organ(img_torch)

    st.success(f"Detected Organ: {organ.upper()} ({organ_conf*100:.2f}%)")

    # STEP 2: Condition detection
    if organ == "lungs":
        st.info("Running Pneumonia Detection...")
        img_chest = preprocess_pneumonia(image)
        label, conf = predict_pneumonia(img_chest)
        st.info(f"Pneumonia Detection: {label} ({conf*100:.2f}%)")

    elif organ == "bone":
        st.info("Running Bone Fracture Detection...")
        img_bone = preprocess_fracture(image)
        label, conf = predict_fracture(img_bone)
        st.info(f"Fracture Detection: {label} ({conf*100:.2f}%)")

    else:
        st.warning(f"No disease model available for: {organ.upper()}")
