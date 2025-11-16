# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from preprocessing import preprocess_image_for_model, make_gradcam_heatmap, overlay_heatmap
import time

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Pneumonia X-ray Detector", page_icon="🩺", layout="centered")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_pneumonia_model(path="xray_cnn_model.h5"):
    model = load_model(path)
    return model

# ----------------------------
# Sidebar - metadata / info
# ----------------------------
st.sidebar.title("Pneumonia X-ray App")
st.sidebar.markdown("**Model:** Keras CNN\n**Input size:** 128×128 (RGB)\n**Classes:** Normal / Pneumonia")
st.sidebar.markdown("---")
st.sidebar.markdown("**Usage:** Upload a chest X-ray image (jpg/jpeg/png).")
st.sidebar.markdown("**Note:** This is a prototype model — not a medical device.")

# Allow user to set model path (optional)
model_path = st.sidebar.text_input("Model path", value="pneumonia_model.h5")
model = load_pneumonia_model(model_path)

# ----------------------------
# App header
# ----------------------------
st.title("🩻 Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray image and get a prediction with confidence. Toggle Grad-CAM for explainability.")

col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = st.file_uploader("Upload chest X-ray image", type=["jpg","jpeg","png"])

with col2:
    st.markdown("**Options**")
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=False)
    confidence_threshold = st.slider("Confidence threshold (%)", min_value=50, max_value=95, value=60, step=5)

# ----------------------------
# Main behavior
# ----------------------------
if uploaded_file is not None:
    # Read image
    try:
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        st.error("Could not read the uploaded image. Try a different file.")
        st.exception(e)
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image_for_model(image, target_size=(128,128))

    # Prediction
    with st.spinner("Predicting..."):
        start = time.time()
        pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
        end = time.time()

    # Interpret result
    # Assuming model was trained with sigmoid binary output: >0.5 => Pneumonia
    pneumonia_prob = float(pred)  # probability of class=1 (Pneumonia)
    normal_prob = 1.0 - pneumonia_prob

    if pneumonia_prob >= 0.5:
        label = "PNEUMONIA"
        conf = pneumonia_prob
    else:
        label = "NORMAL"
        conf = normal_prob

    # Display results
    st.subheader("🔍 Prediction")
    st.markdown(f"**Result:** `{label}`")
    st.progress(int(conf * 100))
    st.write(f"**Confidence:** {conf*100:.2f}%")
    st.write(f"**Prediction time:** {(end-start):.2f} seconds")

    # Confidence threshold message
    if conf*100 < confidence_threshold:
        st.warning(f"Model confidence ({conf*100:.1f}%) is below your threshold ({confidence_threshold}%). Interpret with caution.")

    # Grad-CAM explainability
    if show_gradcam:
        st.markdown("---")
        st.subheader("🧭 Grad-CAM Explanation (approximate)")

        try:
            heatmap = make_gradcam_heatmap(np.expand_dims(img_array, axis=0), model, last_conv_layer_name=None)
            overlay = overlay_heatmap(Image.fromarray((img_array*255).astype("uint8")), heatmap)
            st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
            st.caption("Red regions indicate areas that contributed most to the model's decision.")
        except Exception as e:
            st.error("Grad-CAM failed for this model or image. It requires a standard Keras model with convolutional layers.")
            st.exception(e)

else:
    st.info("Upload a chest X-ray image to get started.")
