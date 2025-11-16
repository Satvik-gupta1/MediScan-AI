
# run_inference.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchxrayvision as xrv
from densenet121_xray_classifier import OrganClassifier  # same class as training

# -------- CONFIG --------
MODEL_PATH = "densenet121_xray_organs.pth"  # your saved model
IMG_PATH = "img.jpg"                 # change this to your image path
CLASS_NAMES = ['bone', 'brain', 'heart', 'lungs']  # must match your training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------

# -------- LOAD MODEL --------
print("Loading trained model...")
base = xrv.models.DenseNet(weights="densenet121-res224-chex")
features = base.features

model = OrganClassifier(features, num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully. Ready for inference.\n")

# -------- IMAGE PREPROCESSING --------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

img = Image.open(IMG_PATH).convert("L")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# -------- INFERENCE --------
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

print(f"🩻 Predicted organ: {pred_class}")
print(f"Confidence: {confidence*100:.2f}%")

# -------- OPTIONAL: visualize result --------
import matplotlib.pyplot as plt
plt.imshow(img, cmap='gray')
plt.title(f"{pred_class} ({confidence*100:.1f}%)")
plt.axis('off')
plt.show()
