from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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
    with open("index.html", "r") as f:
        return f.read()

@app.post("/inference/")
async def inference(data: UploadFile = File(...), model_name: Optional[ItemEnum] = Form(ItemEnum.ViT)):
    """Run inference on image."""
    if not data.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    if model_name.value not in models:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    model = models[model_name.value]

    start = time()
    image_data = await data.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("L")
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    image_tensor = torch.tensor(image_array).unsqueeze(0)

    print("Image loaded in {} seconds".format(time() - start))

    classes = {0: "glioma", 1: "meningioma", 2: "no-tumor", 3: "pituitary"}

    start = time()
    with torch.no_grad():
        print(image_array.size)
        logits = model(image_tensor)
        ps = torch.exp(logits)
        top_p, top_class = ps.topk(1, dim=1)
        prediction = classes[top_class.item()]

    print("Prediction made in {} seconds".format(time() - start))

    # Convert the image to base64 for returning with response
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={"prediction": prediction, "image": img_base64})
