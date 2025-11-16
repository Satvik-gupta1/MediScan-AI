# app.py — Streamlit polished X-Ray Organ Analyzer (Dark Dashboard)
# Requires: streamlit, torch, torchvision, torchxrayvision, pillow, opencv-python, matplotlib

import os
import io
import base64
from typing import Tuple, List, Dict

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

import streamlit as st
import torchxrayvision as xrv

# ---------- CONFIG ----------
CLASS_NAMES = ['bone', 'brain', 'heart', 'lungs']
MODEL_PATH = "densenet121_xray_organs.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

# ---------- Small utils ----------
def emoji_for_conf(p: float) -> str:
    if p >= 0.95: return "🚀"
    if p >= 0.90: return "🔥"
    if p >= 0.80: return "✅"
    if p >= 0.65: return "👍"
    if p >= 0.50: return "⚠️"
    return "❓"

def label_for_conf(p: float) -> str:
    if p >= 0.95: return "Very likely"
    if p >= 0.90: return "Very likely"
    if p >= 0.80: return "Likely"
    if p >= 0.65: return "Possible"
    if p >= 0.50: return "Uncertain"
    return "Unlikely"

# ---------- Model / preprocessing ----------
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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}. Place it next to app.py")
    base = xrv.models.DenseNet(weights="densenet121-res224-chex")
    features = base.features
    model = OrganClassifier(features, num_classes=len(CLASS_NAMES)).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

_preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def prepare_image(pil_img: Image.Image) -> torch.Tensor:
    return _preprocess(pil_img.convert("L")).unsqueeze(0).to(DEVICE)

def predict(model: nn.Module, pil_img: Image.Image) -> Tuple[List[Tuple[str, float]], np.ndarray]:
    inp = prepare_image(pil_img)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pairs = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: x[1], reverse=True)
    return pairs, probs

# ---------- Grad-CAM ----------
def find_last_conv_layer(module: nn.Module):
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found in backbone")
    return last

def grad_cam(model: nn.Module, pil_img: Image.Image, class_index: int = 0, alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    model.zero_grad()
    inp = prepare_image(pil_img)

    activations = []
    gradients = []

    target_conv = find_last_conv_layer(model.backbone)

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_conv.register_forward_hook(forward_hook)
    bh = target_conv.register_full_backward_hook(backward_hook)

    out = model(inp)
    out[0, class_index].backward()

    act = activations[0].detach().cpu().squeeze(0)   # C,H,W
    grad = gradients[0].detach().cpu().squeeze(0)    # C,H,W

    weights = torch.mean(grad, dim=(1,2)).numpy()   # C
    cam = np.zeros((act.shape[1], act.shape[2]), dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i].numpy()

    cam = np.maximum(cam, 0)
    cam_norm = cam / cam.max() if cam.max() != 0 else cam
    cam_resized = cv2.resize(cam_norm, (224, 224))

    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    orig = np.array(pil_img.convert('L').resize((224, 224)))
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap, alpha, 0)

    fh.remove()
    bh.remove()

    return overlay, cam_resized

def heatmap_color_weights(cam_resized: np.ndarray) -> Dict[str, float]:
    bins, labels = 4, ['Blue (low)', 'Green', 'Yellow', 'Red (high)']
    flat = cam_resized.flatten()
    thresholds = np.linspace(0.0, 1.0, bins + 1)
    counts = [np.sum((flat >= thresholds[i]) & (flat < thresholds[i+1])) for i in range(bins)]
    total = flat.size
    return {labels[i]: float(counts[i]) / total for i in range(bins)}

# ---------- Styling (dark medical theme) ----------
DARK_CSS = """
<style>
/* Background and card styles */
[data-testid="stAppViewContainer"] { background-color: #0f1720; }
[data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
[data-testid="stSidebar"] { background-color: #0b1220; }
/* Title */
h1 { color: #e6f0ff !important; font-weight:700; }
/* panel boxes */
.card {
  background: linear-gradient(180deg, #0f1726, #101826);
  border-radius: 12px;
  padding: 14px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.5);
  color: #e6eef8;
}
/* buttons */
.stButton>button {
  background-color:#0ea5a3;
  color: white;
  border-radius: 8px;
  padding: 8px 14px;
}
/* progress bars (container) */
.conf-bar {
  height: 18px;
  background: linear-gradient(90deg,#0f1726,#152433);
  border-radius: 9px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.04);
}
.conf-fill {
  height: 100%;
  border-radius: 9px;
  text-align: right;
  padding-right:8px;
  font-weight:600;
  color: #08121a;
}
/* small muted text */
.small-muted { color: rgba(230,240,255,0.55); font-size:13px; }
.left-col { padding-right: 18px; }
</style>
"""

# ---------- Streamlit UI ----------
def streamlit_app():
    st.set_page_config(page_title="AI X-Ray Organ Analyzer", page_icon="🩻", layout="wide")
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # Header
    c1, c2 = st.columns([0.12, 0.88])
    with c1:
        st.image(Image.new('RGBA', (80,80), (12,18,30,255)), width=70)  # small placeholder (or replace with logo)
    with c2:
        st.markdown("<h1>AI X-Ray Organ Analyzer</h1>", unsafe_allow_html=True)
        st.markdown('<div class="small-muted">DenseNet121 (CheX) • Organ classification • Grad-CAM explainability</div>', unsafe_allow_html=True)

    st.markdown("---")

    model = load_model()

    # Upload area styled big
    uploaded = st.file_uploader("Drag & drop or browse an X-ray image (JPG/PNG)", type=["jpg","jpeg","png"])
    if uploaded is None:
        st.info("Upload a frontal X-ray image to get a prediction and Grad-CAM visualization.")
        return

    # Read image
    pil_img = Image.open(uploaded).convert('L')

    # Prediction + gradcam
    pairs, _ = predict(model, pil_img)
    top_class = pairs[0][0]
    alpha = st.sidebar.slider("Grad-CAM overlay alpha", min_value=0.15, max_value=0.7, value=0.4, step=0.05)
    overlay, cam = grad_cam(model, pil_img, class_index=CLASS_NAMES.index(top_class), alpha=alpha)
    color_weights = heatmap_color_weights(cam)

    # Layout: left confidence column, right image viewer
    left, right = st.columns([0.38, 0.62])

    # Left panel - styled card
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin:0 0 6px 0">Predictions</h3>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Sorted by confidence — top result highlighted.</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        # show each class as a small card with bar
        for idx, (name, prob) in enumerate(pairs):
            # color mapping for fill
            if prob >= 0.8:
                fill_color = "#00d68f"   # green
            elif prob >= 0.65:
                fill_color = "#ffd60a"   # yellow
            elif prob >= 0.5:
                fill_color = "#ff7a59"   # orange
            else:
                fill_color = "#6b7280"   # gray

            descriptor = label_for_conf(prob)
            emoji = emoji_for_conf(prob)
            percent_text = f"{prob*100:.1f}%"

            # highlight top result
            if idx == 0:
                st.markdown(f'<div style="padding:10px;border-radius:10px;background:linear-gradient(90deg,#062022,#0b2433);margin-bottom:10px;">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                            f'<div><strong style="font-size:18px;color:#e6f7ff">{name.capitalize()}</strong><div style="color:#9fb4c8">{descriptor} • {emoji}</div></div>'
                            f'<div style="text-align:right;"><span style="font-weight:700;color:#e6f7ff">{percent_text}</span></div>'
                            f'</div>'
                            f'<div style="height:10px;"></div>'
                            f'<div class="conf-bar"><div class="conf-fill" style="width:{prob*100}%;background:{fill_color};"></div></div>'
                            f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding:8px;border-radius:8px;background:rgba(255,255,255,0.02);margin-bottom:8px;">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                            f'<div><strong style="font-size:15px;color:#dff3ff">{name.capitalize()}</strong><div style="color:#9fb4c8">{descriptor} • {emoji}</div></div>'
                            f'<div style="text-align:right;"><span style="font-weight:700;color:#dff3ff">{percent_text}</span></div>'
                            f'</div>'
                            f'<div style="height:8px;"></div>'
                            f'<div class="conf-bar"><div class="conf-fill" style="width:{prob*100}%;background:{fill_color};"></div></div>'
                            f'</div>', unsafe_allow_html=True)

        # small metadata and download
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Model: DenseNet121 (CheX) • Demo</div>', unsafe_allow_html=True)
        # download overlay
        _, png_buf = cv2.imencode('.png', overlay)
        st.download_button("Download Grad-CAM PNG", data=png_buf.tobytes(), file_name="gradcam_overlay.png", mime="image/png")
        st.markdown('</div>', unsafe_allow_html=True)

    # Right panel - images side-by-side
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<h3 style="margin:0">Images</h3>'
                    f'<div style="color:#9fb4c8;font-size:14px">Top: {top_class.capitalize()} ({pairs[0][1]*100:.1f}%)</div>'
                    '</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        # side-by-side images
        cA, cB = st.columns(2)
        with cA:
            st.markdown("<div style='text-align:center;color:#cfe9ff;font-weight:700;margin-bottom:6px'>Original</div>", unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
        with cB:
            st.markdown("<div style='text-align:center;color:#cfe9ff;font-weight:700;margin-bottom:6px'>Analyzed (Grad-CAM)</div>", unsafe_allow_html=True)
            # convert overlay BGR->RGB for display
            overlay_disp = overlay[:, :, ::-1]
            st.image(overlay_disp, use_container_width=True)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        # color-weight bar chart
        st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;">'
                    '<div style="font-weight:700;color:#dff3ff">Grad-CAM color weight</div>'
                    '<div style="color:#9fb4c8">Blue → Red (low → high)</div>'
                    '</div>', unsafe_allow_html=True)

        labels = list(color_weights.keys())
        vals = [color_weights[k]*100 for k in labels]
        fig, ax = plt.subplots(figsize=(6,2.2), facecolor='#0f1720')
        bars = ax.bar(labels, vals, color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'], alpha=0.9)
        ax.set_ylim(0, 100)
        ax.set_facecolor('#0f1720')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
        ax.set_ylabel('Percent (%)', color='white')
        for spine in ['top','right']: ax.spines[spine].set_visible(False)
        for i, v in enumerate(vals):
            ax.text(i, v + 1.5, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
        st.pyplot(fig, clear_figure=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # small footer
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:#9fb4c8">Made with ❤️  •  DenseNet121-CheX • Author: Savage</div>', unsafe_allow_html=True)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    streamlit_app()
