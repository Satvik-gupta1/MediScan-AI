import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("type_classifier.h5")
IMG_SIZE = (224, 224)

def predict_type(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    if pred < 0.5:
        print("✅ Chest X-ray detected")
    else:
        print("✅ Bone X-ray detected")
predict_type(r"C:\Users\naman\OneDrive\Desktop\Example folder\NORMAL2-IM-0609-0001.jpeg")
