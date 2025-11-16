import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.write("➡ App Started")  # DEBUG

# Debug: show working directory
st.write("Current Directory:", os.getcwd())
st.write("Files:", os.listdir())

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Model
@st.cache_resource
def load_model():
    try:
        st.write("➡ Loading model...")  # DEBUG
        model = tf.keras.models.load_model("bone_fracture_model.h5", compile=False)
        st.write("✔ Model Loaded Successfully")
        return model
    except Exception as e:
        st.error(f"❌ Model Load Failed: {e}")
        return None

model = load_model()

def predict_fracture(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return (("🦴 **Fractured**",1-pred) if pred <= 0.5 else ("✅ **Not Fractured**",pred))

st.title("🩻 Bone Fracture Detection App")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Analyze"):
        if model is None:
            st.error("Model unavailable ❌")
        else:
            with st.spinner("Analyzing..."):
                label, prob = predict_fracture(image)

            st.subheader("Result:")
            st.markdown(label)
            st.write("Score:", prob)
