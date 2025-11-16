# utils/preprocessing.py
import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image_for_model(pil_image, target_size=(128,128)):
    """
    Convert PIL image -> normalized numpy array suitable for model.predict
    """
    img = pil_image.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    # if grayscale image was provided, ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    return arr

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Compute Grad-CAM heatmap for a Keras model.
    - img_array: numpy array (1, H, W, C) normalized to [0,1]
    - model: loaded Keras model
    - last_conv_layer_name: optional name; if None we try to infer
    Returns: heatmap (H,W) normalized 0..1
    """
    # Try to infer last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No Conv2D layer found in the model to compute Grad-CAM.")

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Assuming binary sigmoid output: get the scalar prediction
        if predictions.shape[-1] == 1:
            pred_index = 0
            class_channel = predictions[:, pred_index]
        else:
            # if model has softmax multi-class, use argmax
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-10
    heatmap /= max_val
    heatmap = heatmap.numpy()
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (img_array.shape[1], img_array.shape[2])).numpy()
    heatmap = heatmap.squeeze()
    return heatmap

def overlay_heatmap(pil_img, heatmap, alpha=0.4, colormap='jet'):
    """
    Overlay heatmap on top of PIL image and return a new PIL image.
    pil_img: PIL.Image RGB
    heatmap: 2D array same size as pil_img
    """
    import matplotlib.cm as cm
    import numpy as np
    img = np.array(pil_img).astype("uint8")
    jet = cm.get_cmap(colormap)
    heatmap_colored = (jet(heatmap)[:,:,:3] * 255).astype("uint8")
    overlayed = (img * (1 - alpha) + heatmap_colored * alpha).astype("uint8")
    return Image.fromarray(overlayed)