# CharClassifier — DevaVision

A custom CNN for **62-class handwritten Devanagari/Sanskrit character recognition**, deployed as an interactive web application with both drawing and image-upload input.

**🔗 Live Demo:** [char-classifier.vercel.app](https://char-classifier.vercel.app/)
**📄 API Docs:** [sanskritchar-api.onrender.com/docs](https://sanskritchar-api.onrender.com/docs)

Draw a character or upload an image → the frontend sends it to a deployed FastAPI service → the CNN classifies it → the app returns the predicted class, confidence score, and top-3 predictions.

---

## Table of Contents

- [Overview](#overview)
- [ML Pipeline](#ml-pipeline)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [CNN Architecture](#cnn-architecture)
- [Experiments](#experiments)
- [Evaluation](#evaluation)
- [Error Analysis](#error-analysis)
- [API](#api)
- [Web Application](#web-application)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

Handwritten characters vary significantly in stroke thickness, shape, slant, and positioning. The goal of this project was not just to train a classifier, but to build and evaluate a **complete deep learning pipeline** — studying how regularization, augmentation, and learning-rate selection affect generalization.

The model was built **from scratch using PyTorch**, rather than relying on transfer learning, to keep the architecture and optimization process fully transparent.

---

## ML Pipeline

```
Dataset → EDA → Preprocessing → Stratified 80/10/10 Split → Custom CNN
   → Baseline → BatchNorm + Dropout → Data Augmentation → Learning-Rate Sweep
   → Final Model → Test Evaluation → Confusion Matrix + Error Analysis
   → Model Export → FastAPI Inference API → Interactive Web Application
```

---

## Dataset

The project uses a handwritten Devanagari/Sanskrit character dataset containing **62 classes**.

Before training, the dataset was analyzed for:
- Class distribution
- Image dimensions and modes
- Pixel value ranges
- Representative character samples

The data was split using stratified sampling:

| Split | Proportion |
|---|---|
| Training | 80% |
| Validation | 10% |
| Test | 10% |

The test set was kept completely separate from model selection and hyperparameter experiments.

---

## Preprocessing

```
Image → Grayscale → Resize (32×32) → Tensor → Normalization
```

Images are converted to grayscale, resized to 32×32, converted to tensors, and normalized using statistics computed from the training data. Training data additionally uses augmentation, while validation and test data use deterministic preprocessing.

---

## CNN Architecture

```
Input: 1 × 32 × 32
   │
   ▼
Conv 1 → 32 channels → BatchNorm + ReLU + MaxPool
   │
   ▼
Conv 2 → 64 channels → BatchNorm + ReLU + MaxPool
   │
   ▼
Conv 3 → 128 channels → BatchNorm + ReLU
   │
   ▼
Global Average Pooling
   │
   ▼
Dropout
   │
   ▼
Linear Layer → 62 Classes
```

The network progressively increases feature depth (32 → 64 → 128), allowing early layers to learn low-level stroke patterns while deeper layers capture character-specific visual features. Global Average Pooling keeps the classification head compact and reduces parameter count compared to directly flattening the feature maps.

---

## Experiments

The model was developed through controlled experiments rather than selecting a single architecture immediately. The final configuration was chosen using validation performance.

| Experiment | Configuration | Purpose |
|---|---|---|
| Baseline | CNN | Establish reference performance |
| Regularized | + BatchNorm + Dropout | Study regularization |
| Augmented | + Data Augmentation | Study robustness to input variation |
| LR Sweep | 3 learning rates | Study optimization behaviour |

**Best Validation Accuracy: 95.35%**

The best checkpoint was selected based on validation accuracy rather than training accuracy. The held-out test set was not used during model selection.

---

## Evaluation

The final model is evaluated using:
- Accuracy
- Macro Precision / Recall / F1
- Confusion Matrix
- Misclassified sample analysis

Macro-averaged metrics are reported because the problem contains 62 classes, and overall accuracy alone does not show how performance varies across individual classes.

| Metric | Score |
|---|---|
| Test Accuracy | **95.45%** |
| Macro Precision | **96.03%** |
| Macro Recall | **95.45%** |
| Macro F1 | **95.36%** |

---

## Error Analysis

A row-normalized confusion matrix is used to identify frequently confused character pairs. Misclassified samples are also visualized with:
- Actual class
- Predicted class
- Prediction confidence

This provides a qualitative view of where the model struggles and connects model errors to visually similar character patterns. Results and visualizations are available in [`results/`](./results).

---

## API

The trained model is served through a FastAPI REST API.

**Swagger Documentation:** https://sanskritchar-api.onrender.com/docs

### Health Check

```
GET /health
```

```json
{
  "status": "ok",
  "num_classes": 62
}
```

### Prediction

```
POST /predict
```

Accepts an image via `multipart/form-data`.

```bash
curl -X POST https://sanskritchar-api.onrender.com/predict \
  -F "file=@character.png"
```

**Example response:**

```json
{
  "predicted_class": "60_8",
  "confidence": 0.9241,
  "top3": [
    { "class_name": "60_8", "confidence": 0.9241 },
    { "class_name": "25_tta", "confidence": 0.0512 },
    { "class_name": "32_da", "confidence": 0.0119 }
  ]
}
```

The inference pipeline uses the same grayscale conversion, resizing, and normalization used during training. Normalization statistics and class names are stored with the model checkpoint so the deployed API can reconstruct the training-time preprocessing configuration.

---

## Web Application

The frontend provides two input modes:

**Draw**
```
Canvas → PNG → FastAPI → CNN → Prediction
```
Users draw a character directly on a canvas.

**Upload**
```
Image → FastAPI → CNN → Prediction
```
Users upload an existing character image.

The interface displays the predicted class, confidence score, and top-3 predictions.

---

## System Architecture

```
                        User
                         │
                  Draw / Upload
                         │
                         ▼
                  ┌─────────────┐
                  │  Frontend   │
                  │   Vercel    │
                  └──────┬──────┘
                         │
                    POST /predict
                         │
                         ▼
                  ┌─────────────┐
                  │   FastAPI   │
                  │   Render    │
                  └──────┬──────┘
                         │
                    Preprocessing
                         │
                         ▼
                  ┌─────────────┐
                  │  Custom CNN │
                  │   PyTorch   │
                  └──────┬──────┘
                         │
                         ▼
              Prediction + Confidence
```

---

## Tech Stack

<p>
  <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/-Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/-Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/-Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/-Uvicorn-2A308B?style=for-the-badge&logo=gunicorn&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/-HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" />
  <img src="https://img.shields.io/badge/-CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" />
  <img src="https://img.shields.io/badge/-JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" />
  <img src="https://img.shields.io/badge/-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" />
  <img src="https://img.shields.io/badge/-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/-Git-F05032?style=for-the-badge&logo=git&logoColor=white" />
</p>

| Layer | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, Torchvision |
| Data Processing | NumPy, Pandas |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Image Processing | Pillow |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Vercel, Render |
| Development | Jupyter Notebook, Git |

---

## Project Structure

```
CharClassifier/
│
├── api/
│   └── main.py
│
├── data/
│   └── Sanskrit Mnist/
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   └── 01_char_recognition.ipynb
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── error_analysis.png
│   └── experiment_comparison.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running Locally

**1. Clone the repository**
```bash
git clone <repo-url>
cd CharClassifier
```

**2. Create a virtual environment**
```bash
python -m venv .venv
```

Activate it:
```bash
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the API**
```bash
uvicorn api.main:app --reload
```

- API: http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

---

## Key Design Decisions

**Custom CNN**
Implemented from scratch to make the architecture, feature extraction, and optimization process explicit rather than hiding them behind a pre-trained backbone.

**Stratified splitting**
Preserves class proportions across training, validation, and test sets.

**Controlled experiments**
Structured so that performance changes could be attributed to specific techniques — regularization, augmentation, or learning-rate selection.

**Validation-based model selection**
The validation set was used for model and hyperparameter selection while the test set remained untouched until final evaluation.

**Train-only augmentation**
Augmentation is applied only to training images. Validation and test images use deterministic preprocessing.

**Checkpoint metadata**
The model checkpoint stores class names, image size, and normalization statistics so the deployment pipeline can reproduce the training-time preprocessing configuration.

## Limitations

- Trained on a single dataset; may not generalize to handwriting styles or pens/pencils outside its distribution.
- Fixed 32×32 grayscale input may lose fine stroke detail, contributing to some class confusions.
- Training set is not perfectly class-balanced; no explicit reweighting was applied.
- Classifies one character at a time — no support for words or connected script.
- Not evaluated on rotated/skewed inputs or out-of-distribution samples.
- No automated tests yet; a smoke test for the `/predict` endpoint is the planned next step.
- Not load-tested for concurrent traffic or large images.

