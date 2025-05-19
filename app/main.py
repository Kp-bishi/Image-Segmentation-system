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
