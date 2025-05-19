#!/bin/bash

echo "📁 Creating directory structure..."
mkdir -p app
mkdir -p frontend

echo "📄 Creating backend files..."
cat <<EOF > app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import load_model
from PIL import Image
import io
import torch

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    # TODO: replace with your actual preprocess function
    tensor = torch.tensor([])  # dummy
    output = model(tensor.unsqueeze(0))
    return {"result": "mask_placeholder"}
EOF

cat <<EOF > app/model.py
from .unet import UNet
import torch
import os

def load_model():
    model = UNet(in_channels=3, out_channels=4)
    if os.path.exists("unet_model.pth"):
        model.load_state_dict(torch.load("unet_model.pth", map_location="cpu"))
    else:
        print("⚠️ unet_model.pth not found. Using untrained model.")
    model.eval()
    return model
EOF

cat <<EOF > app/unet.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
EOF

cat <<EOF > requirements.txt
fastapi
uvicorn
torch
pillow
EOF

echo "✅ Project skeleton created. Now:"
echo "1. Upload your trained model to: unet_model.pth"
echo "2. Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
