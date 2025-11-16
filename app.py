# app.py — Compact Dark UI (Logic Unchanged)

import os
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchxrayvision as xrv

import streamlit as st
from tensorflow.keras.models import load_model as keras_load_model
from util.preprocessing import preprocess_image_for_model

# ---------------------------- CONFIG ----------------------------
CLASS_NAMES = ['bone', 'brain', 'heart', 'lungs']
MODEL_PATH = "densenet121_xray_organs.pth"
PNEUMONIA_MODEL_PATH = "pneumonia_model.h5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------

# ------------------------- Helper Utils -------------------------
def emoji_for_conf(p):
    if p >= 0.95: return "🚀"
    if p >= 0.90: return "🔥"
    if p >= 0.80: return "✅"
    if p >= 0.65: return "👍"
    if p >= 0.50: return "⚠️"
    return "❓"

# ------------------------ Model Definition -----------------------
class OrganClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

@st.cache_resource
def load_model():
    base = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model = OrganClassifier(base.features, num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_pneumonia_model_tf():
    return keras_load_model(PNEUMONIA_MODEL_PATH)

# --------------------- Image Preprocessing -----------------------
_preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def prepare_image(pil_img):
    return _preprocess(pil_img.convert("L")).unsqueeze(0).to(DEVICE)

# ------------------------ Prediction Logic -----------------------
def predict(model, pil_img):
    inp = prepare_image(pil_img)
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0].cpu().numpy()
    pairs = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: x[1], reverse=True)
    return pairs, probs

# ---------------------- Pneumonia Logic --------------------------
def predict_pneumonia(pil_img):
    model = load_pneumonia_model_tf()
    img_arr = preprocess_image_for_model(pil_img, target_size=(128,128))
    img_arr = np.expand_dims(img_arr, 0)
    prob = float(model.predict(img_arr)[0][0])
    return ("Pneumonia Detected", prob) if prob > 0.5 else ("Normal Lungs", 1 - prob)

# -------------------------- Grad-CAM -----------------------------
def find_last_conv_layer(module):
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d): last = m
    return last


def grad_cam(model, pil_img, class_index, alpha=0.4):
    model.zero_grad()

    inp = prepare_image(pil_img)
    activ, grads = [], []
    layer = find_last_conv_layer(model.backbone)

    def f_hook(m, i, o): activ.append(o)
    def b_hook(m, gi, go): grads.append(go[0])

    fh = layer.register_forward_hook(f_hook)
    bh = layer.register_full_backward_hook(b_hook)

    out = model(inp)
    out[0, class_index].backward()

    act = activ[0].detach().cpu().squeeze(0)
    grad = grads[0].detach().cpu().squeeze(0)

    w = torch.mean(grad, dim=(1,2)).numpy()
    cam = np.maximum(np.sum(w[:,None,None] * act.numpy(), axis=0), 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    cam = cv2.resize(cam, (224,224))

    heat = np.uint8(255 * cam)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    orig = np.array(pil_img.convert('L').resize((224,224)))
    orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(orig, 1-alpha, heat, alpha, 0)

    fh.remove(); bh.remove()
    return overlay

# ---------- Pneumonia Grad-CAM ----------
def grad_cam_pneumonia(model, pil_img, alpha=0.35):
    # Preprocess to 128x128 + 3-channel
    img_arr = preprocess_image_for_model(pil_img, target_size=(128, 128))
    img_arr = np.expand_dims(img_arr, 0)

    # Find last conv layer in Keras model
    last_conv = None
    for layer in reversed(model.layers):
        if "conv" in layer.name and "Conv2D" in str(layer.__class__):
            last_conv = layer.name
            break
    if last_conv is None:
        raise ValueError("No Conv2D layer found in pneumonia model.")

    grad_model = keras_load_model.__globals__['tf'].keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    import tensorflow as tf
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_arr)
        pred_value = preds[0][0]  # sigmoid output

    grads = tape.gradient(pred_value, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-9

    # Resize heatmap to original X-ray size
    heatmap = cv2.resize(heatmap, (300, 300))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Original resized
    orig = cv2.cvtColor(np.array(pil_img.resize((300, 300))), cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(orig, 1 - alpha, heatmap_color, alpha, 0)
    return overlay, float(pred_value)

# ----------------------------- UI -------------------------------
DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background-color:#0f1720; }
h1 { color:#e6f0ff;font-weight:700; }
.card { background:#141c27;padding:15px;border-radius:12px; }
.pred-box {background:#16212e;padding:10px;border-radius:10px;margin-bottom:8px;color:#dbe9ff;}
</style>
"""


# ------------------------ Streamlit App --------------------------
def streamlit_app():
    st.set_page_config(page_title="AI X-Ray Analyzer", layout="wide")
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    st.markdown("<h1>AI X-Ray Organ Analyzer</h1>", unsafe_allow_html=True)

    model = load_model()

    uploaded = st.file_uploader("Upload X-ray Image", type=["jpg","png","jpeg"])
    if uploaded is None:
        st.info("Upload a chest X-ray to begin.")
        return

    pil_img = Image.open(uploaded).convert("L")
    pairs, _ = predict(model, pil_img)
    top_class = pairs[0][0]

    overlay = grad_cam(model, pil_img, CLASS_NAMES.index(top_class))

    # Original (convert PIL → numpy)
    orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2RGB)
    orig_img = cv2.resize(orig_img, (300, 300))

    # Grad-CAM overlay (already numpy BGR)
    overlay_img = cv2.resize(overlay[:, :, ::-1], (300, 300))

    col1, col2 = st.columns(2)

    with col1:
        st.image(orig_img, caption="Original", use_container_width=False)

    with col2:
        st.image(overlay_img, caption=f"Grad-CAM ({top_class})", use_container_width=False)


    # ---------------- SIDE PANEL CONFIDENCE ----------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'><b>Prediction Confidence</b></div>", unsafe_allow_html=True)

    for organ, prob in pairs:
        st.markdown(f"<div class='pred-box'><b>{organ.capitalize()}</b> — {prob*100:.2f}% {emoji_for_conf(prob)}</div>", unsafe_allow_html=True)

    # ---------------- Pneumonia Check ----------------
    if top_class == "lungs":
        st.warning("Detected: Lungs. You can run Pneumonia Detection.")
        if st.button("Run Pneumonia Detection"):
            label, prob = predict_pneumonia(pil_img)
            if label == "Pneumonia Detected":
                st.error(f"⚠️ {label} — {prob*100:.2f}%")
            else:
                st.success(f"✅ {label} — {(prob)*100:.2f}%")


# -------------------------- Entrypoint ---------------------------
if __name__ == "__main__":
    streamlit_app()