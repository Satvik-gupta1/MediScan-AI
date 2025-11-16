import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchxrayvision as xrv
from densenet121_xray_classifier import OrganClassifier

# -------- CONFIG --------
MODEL_PATH = "densenet121_xray_organs.pth"
IMG_PATH = "test_image.jpg"
CLASS_NAMES = ['bone', 'brain', 'heart', 'lungs']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------

# -------- LOAD MODEL --------
base = xrv.models.DenseNet(weights="densenet121-res224-chex")
features = base.features
model = OrganClassifier(features, num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------- IMAGE PREPROCESSING --------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

img = Image.open(IMG_PATH).convert("L")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# -------- GRAD-CAM IMPLEMENTATION --------
# Try to find the last conv layer of DenseNet backbone
target_layer = None
for name, module in model.backbone.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        target_layer = module  # keep updating, last one stays

print(f"Using target layer for Grad-CAM: {target_layer}")


# Hooks to capture gradients and activations
gradients = []
activations = []

def save_gradient(grad):
    gradients.append(grad)

def forward_hook(module, input, output):
    activations.append(output)

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(lambda m, gin, gout: save_gradient(gout[0]))

# Forward pass
output = model(input_tensor)
probs = torch.softmax(output, dim=1)
pred_idx = torch.argmax(probs).item()
pred_class = CLASS_NAMES[pred_idx]

# Backward pass
model.zero_grad()
output[0, pred_idx].backward()

# Extract grads/activations
grads = gradients[0].cpu().data.numpy()[0]
acts = activations[0].cpu().data.numpy()[0]

weights = np.mean(grads, axis=(1, 2))
cam = np.zeros(acts.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * acts[i, :, :]

# Normalize CAM
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam / cam.max()

# Convert image for display
img_np = np.array(img.resize((224, 224)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_np, cmap='gray')
plt.title(f"Original Image\nPrediction: {pred_class}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Grad-CAM Visualization")
plt.axis('off')

plt.tight_layout()
plt.show()
