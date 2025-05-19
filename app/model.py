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
