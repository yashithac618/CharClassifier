import io
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn, pool):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self, num_classes, use_batchnorm=True, use_dropout=True, dropout_p=0.3):
        super().__init__()
        self.block1 = ConvBlock(1, 32, use_batchnorm, pool=True)
        self.block2 = ConvBlock(32, 64, use_batchnorm, pool=True)
        self.block3 = ConvBlock(64, 128, use_batchnorm, pool=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p) if use_dropout else nn.Identity()
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_PATH = BASE_DIR / "models" / "best_model.pth"

_ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
_class_names = _ckpt["class_names"]
_image_size = _ckpt["image_size"]
_mean = float(_ckpt["normalize_mean"])
_std = float(_ckpt["normalize_std"])

_model = CustomCNN(
    num_classes=len(_class_names),
    use_batchnorm=_ckpt.get("use_batchnorm", True),
    use_dropout=_ckpt.get("use_dropout", True),
    dropout_p=_ckpt.get("dropout_p", 0.3)
)
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()

_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((_image_size, _image_size)),
    transforms.ToTensor(),
    transforms.Normalize((_mean,), (_std,))
])

app = FastAPI(
    title="CharClassifier API",
    description="Handwritten Sanskrit character recognition using a CNN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your Vercel domain before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "num_classes": len(_class_names)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("L")
    x = _transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = _model(x)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idx = torch.topk(probs, k=min(3, len(_class_names)))

    return {
        "predicted_class": _class_names[int(top_idx[0])],
        "confidence": float(top_probs[0]),
        "top3": [
            {"class_name": _class_names[int(i)], "confidence": float(p)}
            for p, i in zip(top_probs, top_idx)
        ]
    }
