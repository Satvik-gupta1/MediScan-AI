MediScan AI: Automated X-ray Analysis (Combines medical context with AI, implies scanning)

# Current Updates
# AI X-Ray Organ Analyzer

This project implements a deep learning system that classifies medical X-ray images by organ type — **bone**, **brain**, **heart**, and **lungs** — and uses Grad-CAM to visualize which regions of the image influence each prediction.
It combines **transfer learning** with **explainable AI** techniques to make medical image classification both accurate and interpretable.

---

## Overview

The model is built on **DenseNet121 pretrained on CheXNet**, a network originally trained for chest X-ray diagnosis.
It is fine-tuned here for organ classification using a **two-stage training process**:

1. **Stage 1 — Head Training:**
   Only the final classifier (the “head”) is trained while keeping the pretrained DenseNet layers frozen.
2. **Stage 2 — Fine-Tuning:**
   All layers are unfrozen and trained with a smaller learning rate, allowing the model to adapt more closely to the new dataset.

A **Streamlit dashboard** provides an interactive interface to upload an X-ray image, view classification probabilities, and visualize model reasoning through Grad-CAM heatmaps.

---

## Key Features

* Pretrained **DenseNet121 (CheXNet)** model for medical transfer learning
* **Organ-level classification:** bone, brain, heart, lungs
* **Explainability through Grad-CAM visualizations**
* **Modern Streamlit dashboard** with clear, dark theme and responsive layout
* **Confidence visualization** with bars, labels, and Grad-CAM overlay
* **Optional Flask API** for integration with external systems

---

## Project Structure

```
AI_Xray_Analyzer/
│
├── app.py                        # Streamlit dashboard and Flask API
├── densenet121_xray_classifier.py # Training script
├── test_single_image.py          # Single image test script
├── grad_cam.py                   # Grad-CAM visualization logic
├── requirements.txt              # Dependencies
├── .gitignore                    # Ignored files (dataset, weights, cache)
├── README.md                     # Project overview
└── dataset/                      # Not included in repository
    ├── train/
    │   ├── bone/
    │   ├── brain/
    │   ├── heart/
    │   └── lungs/
    ├── val/
    │   ├── bone/
    │   ├── brain/
    │   ├── heart/
    │   └── lungs/
    └── test/
        ├── bone/
        ├── brain/
        ├── heart/
        └── lungs/
```

Each of the `train`, `val`, and `test` folders contains **four subfolders** corresponding to the organ categories — `bone`, `brain`, `heart`, and `lungs`.
The model learns to distinguish between these classes using labeled X-ray images in this structure.

---

## How It Works

1. **Data Preparation**
   Organ X-ray images are placed in the folder structure above, separated by organ type.

2. **Training**

   * **Stage 1:** The DenseNet base is frozen, and only the classifier head is trained.
   * **Stage 2:** All layers are unfrozen for fine-tuning on the dataset.

3. **Evaluation and Visualization**
   After training, the model predicts organ types on unseen X-rays and generates **Grad-CAM overlays** to highlight important image regions.

---

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

### 3. (Optional) Run the Flask API

```bash
python app.py flask
```

### 4. Upload and Analyze

Use the Streamlit interface to:

* Upload an X-ray image
* View organ classification probabilities
* Inspect Grad-CAM visualizations
* Download the Grad-CAM overlay image

---

## Model Details

| Parameter                 | Description                      |
| ------------------------- | -------------------------------- |
| Base Model                | DenseNet121 (CheXNet pretrained) |
| Input Size                | 224 × 224 grayscale              |
| Optimizer                 | Adam                             |
| Loss                      | Cross-Entropy                    |
| Batch Size                | 16                               |
| Learning Rate (head)      | 1e-3                             |
| Learning Rate (fine-tune) | 1e-5                             |
| Epochs (head)             | 5                                |
| Epochs (fine-tune)        | 10                               |

---


Total images: ~3,836 images |
Classes: Heart, Lungs, Bone, Brain
| Format: JPEG images resized to 150×150 / 224×224

---
