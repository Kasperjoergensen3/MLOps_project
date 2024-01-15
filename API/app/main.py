from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from enum import Enum
import shutil
import numpy as np
import torch
from omegaconf import OmegaConf
from src.utilities.modules import recursive_find_python_class
from PIL import Image
from typing import Optional
import io
import base64
from time import time
from src.data.make_dataset import transform_image

class ItemEnum(Enum):
    ViT = "ViT"
    SimpleCNN = "SimpleCNN"

def load_models():
    models = {}
    for model_name in ItemEnum:
        model_path = Path("models/test_predict/{}".format(model_name.value))
        config_path = model_path.joinpath(".hydra", "config.yaml")
        checkpoint_path = model_path.joinpath("checkpoints", "best-checkpoint.ckpt")
        config = OmegaConf.load(config_path)
        module = recursive_find_python_class(config.model.name)
        model = module(config)
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        models[model_name.value] = model
    return models

app = FastAPI()
start = time()
models = load_models()
print("Model loaded in {} seconds".format(time() - start))

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("API/app/app.html", "r") as f:
        return f.read()

@app.post("/inference/")
async def inference(request: Request):
    form_data = await request.form()
    data = form_data["data"]
    model_name = form_data["model"]
    if not data.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    if model_name not in models:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    model = models[model_name]

    start = time()
    image_data = await data.read()
    image = Image.open(io.BytesIO(image_data))
    image = transform_image(image)

    print("Image loaded in {} seconds".format(time() - start))

    classes = {0: "glioma", 1: "meningioma", 2: "no-tumor", 3: "pituitary"}

    start = time()
    with torch.no_grad():
        print(image_tensor.size())
        logits = model(image_tensor)
        ps = torch.exp(logits)
        top_p, top_class = ps.topk(1, dim=1)
        prediction = classes[top_class.item()]

    print("Prediction made in {} seconds".format(time() - start))

    # Convert the image to base64 for returning with response
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={"prediction": prediction, "image": img_base64, "model_name": model_name})
