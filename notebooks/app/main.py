from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
from fastapi import UploadFile, File, HTTPException
from typing import Optional
from pathlib import Path
from src.utilities.modules import recursive_find_python_class
from omegaconf import OmegaConf
import torch
import numpy as np
from PIL import Image
import io
from fastapi.responses import HTMLResponse
import base64

class ItemEnum(Enum):
    ViT = "ViT"
    SimpleCNN = "SimpleCNN"

app = FastAPI()

@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.post("/inference/")
async def inference(
    data: UploadFile = File(...), model: Optional[ItemEnum] = ItemEnum.ViT
):
    """Run inference on image."""
    if not data.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    model_path = Path("models/test_predict")
    config_path = model_path.joinpath(".hydra", "config.yaml")
    checkpoint_path = model_path.joinpath("checkpoints", "best-checkpoint.ckpt")
    config = OmegaConf.load(config_path)
    module = recursive_find_python_class(config.model.name)
    model = module(config)
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval() 

    image_data = await data.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("L")
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    image_tensor = torch.tensor(image_array).unsqueeze(0)

    classes = {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4"}

    with torch.no_grad():
        print(image_array.size)
        logits = model(image_tensor)
        ps = torch.exp(logits)
        top_p, top_class = ps.topk(1, dim=1)
        prediction = classes[top_class.item()]

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    # Create a HTML response
    html_content = f"""
    <html>
        <body>
            <h2>Prediction: {prediction}</h2>
            <img src="data:image/jpeg;base64,{img_str}" />
        </body>
    </html>
    """

    response = {
        "model": config.model.name,
        "prediction": prediction
    }
  
    return HTMLResponse(content=html_content)

