# CharClassifier

A custom CNN that classifies handwritten Devanagari/Sanskrit characters across **62 classes**, deployed as a live web app with draw and upload input.

![Status](https://img.shields.io/badge/status-live-brightgreen?style=flat-square)

**Live demo:** https://char-classifier-6dsclr1hv-yashitha618-1281s-projects.vercel.app/

Draw a character or upload an image → it's sent to a FastAPI backend → classified by the trained CNN → returns the predicted class with confidence and top-3 alternatives.

---

## Problem

Handwritten characters vary a lot — stroke thickness, slant, positioning — even for the same class. The goal was to build a CNN from scratch (not a pre-trained model) and *study* how architecture choices, regularization, and augmentation each affect generalization, rather than just chasing one final accuracy number.

## Pipeline

```
Dataset → EDA → Preprocessing → Stratified 80/10/10 Split
   → Custom CNN → [Baseline | +BatchNorm/Dropout | +Augmentation]
   → LR sweep → Final model → Test evaluation → Error analysis
   → Model export → FastAPI → Web app (Vercel + Render)
```

## Model

```
Input 1×32×32
  → Conv(32) + BN + ReLU + MaxPool
  → Conv(64) + BN + ReLU + MaxPool
  → Conv(128) + BN + ReLU
  → Global Average Pool → Dropout → Linear → 62 classes
```

Channel depth increases progressively (32→64→128) so early layers pick up simple strokes and deeper layers learn character-level structure. Global Average Pooling keeps the classifier head small and less prone to overfitting than a large flattened FC layer.

## Experiments

Each stage changes **one variable** at a time, so gains can be attributed to a specific technique rather than several changes at once:

| Stage | Change | Purpose |
|---|---|---|
| Baseline | Plain CNN | Reference point |
| + Regularization | BatchNorm + Dropout | Effect of regularization alone |
| + Augmentation | Same model, augmented data | Effect of data augmentation alone |
| LR sweep | Best config, 3 learning rates | Isolate optimizer tuning |

**Best validation accuracy: 95.35%**, selected by val accuracy (not train accuracy) to avoid picking an overfit checkpoint. Test set stayed untouched until final evaluation.

## Evaluation

Reported with macro-averaged Precision/Recall/F1 (not just accuracy), since with 62 classes, overall accuracy can mask poor performance on individual classes.

| Metric | Score |
|---|---|
| Test Accuracy | *TBD* |
| Macro Precision | *TBD* |
| Macro Recall | *TBD* |
| Macro F1 | *TBD* |

A confusion matrix and misclassified-example grid (actual vs. predicted vs. confidence) are used to identify which characters get confused and why — see `results/`.

## API

**Interactive docs (Swagger):** `https://sanskritchar-api.onrender.com/docs`

### `GET /health`
Returns service status and the number of classes the model was trained on.

```bash
curl https://sanskritchar-api.onrender.com/health
```
```json
{ "status": "ok", "num_classes": 62 }
```

### `POST /predict`
Accepts a single image file (`multipart/form-data`), returns the predicted class, its confidence, and the top-3 predictions.

```bash
curl -X POST https://sanskritchar-api.onrender.com/predict \
  -F "file=@character.png"
```
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

Preprocessing (grayscale → resize → normalize) is identical at inference and training time — the normalization stats and class list are loaded directly from `best_model.pth`, so the two can't silently drift apart.

## Architecture

```
User → Draw/Upload → Frontend (Vercel) → POST /predict
     → FastAPI (Render) → Custom CNN (PyTorch) → Prediction + confidence
```

## Tech Stack

| Layer | Technology |
|---|---|
| Modeling | PyTorch, Torchvision, scikit-learn, NumPy, Pandas |
| Backend | FastAPI, Uvicorn, Python 3.10 |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Vercel (frontend), Render (backend) |
| Tooling | Git, Jupyter

## Project Structure

```
CharClassifier/
├── api/main.py              # FastAPI inference service
├── frontend/                 # Draw/upload web UI
├── models/best_model.pth     # Trained weights + metadata
├── notebooks/                # EDA → training → evaluation
├── results/                  # Confusion matrix, curves, error analysis
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone <repo-url>
cd CharClassifier
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn api.main:app --reload
```
API: `http://127.0.0.1:8000` · Docs: `http://127.0.0.1:8000/docs`

## Key Design Decisions

- **Custom CNN over transfer learning** — keeps the architecture and learning process fully explicit and analyzable.
- **Stratified split** — class proportions preserved across train/val/test.
- **Validation-based model selection** — test set stays unbiased until the very end.
- **Train-only augmentation** — val/test always evaluated on clean, deterministic preprocessing.
- **Metadata saved with checkpoint** — normalization stats and class names travel with the model, so training and deployment can't drift apart.
