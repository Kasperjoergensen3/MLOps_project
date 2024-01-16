import io
import base64

import numpy as np
import torch
import pandas as pd
import pickle
from fastapi import FastAPI, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from enum import Enum
from omegaconf import OmegaConf
from PIL import Image
from time import time
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from prometheus_fastapi_instrumentator import Instrumentator
from google.cloud import storage

from src.data.make_dataset import transform_image
from src.utilities.modules import recursive_find_python_class

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

def add_feature_vec_to_DB(now:str, img:torch.Tensor, class_pred:str):
    mean = torch.mean(img)
    contrast = torch.std(img)
    # Initialize the client
    BUCKET_NAME = "api_user_inputs"
    MODEL_FILE = "input.csv"
    client = storage.Client(project='mlopsproject')
    bucket = client.get_bucket(BUCKET_NAME)

    # Download the file
    blob = bucket.blob(MODEL_FILE)
    blob.download_to_filename(MODEL_FILE)

    #Add feature to database
    with open(MODEL_FILE, "a") as file:
        file.write(f"\n{now}, {mean}, {contrast}, {class_pred}")

    #Upload file to GCP
    blob.upload_from_filename(MODEL_FILE)
    return 

client = storage.Client(project='mlopsproject')
data_bucket = client.get_bucket("data_splits_group29")
blob_data = data_bucket.blob("data/processed/train_features.csv")
blob_data.download_to_filename("data/processed/train_features.csv")
train_data = pd.read_csv("data/processed/train_features.csv")

def data_drift_func(): 
    client = storage.Client(project='mlopsproject')
    bucket = client.get_bucket("api_user_inputs")

    # Download the file
    blob = bucket.blob("input.csv")
    blob.download_to_filename("API/app/input.csv")
    inf_data = pd.read_csv("API/app/input.csv")
    inf_data.drop(["time"], axis = 1, inplace=True)

    #Generate report
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    report.run(reference_data=train_data, current_data=inf_data)
    report.save_html('API/app/report.html')

    #Upload report to GCP
    blob = bucket.blob("report.html")
    blob.upload_from_filename("API/app/report.html")
    return

app = FastAPI()
counter = 0
start = time()
models = load_models()
print("Model loaded in {} seconds".format(time() - start))

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("API/app/app.html", "r") as f:
        return f.read()
    
@app.get("/drift-report", response_class=HTMLResponse)
async def read_drift_report():
    client = storage.Client(project='mlopsproject')
    bucket = client.get_bucket("api_user_inputs")

    # Download the file
    blob = bucket.blob("report.html")
    blob.download_to_filename("API/app/drift_report.html")
    with open("API/app/drift_report.html", "r", encoding='utf-8') as f:
        html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)

@app.post("/inference/")
async def inference(request: Request, background_tasks: BackgroundTasks):
    global counter 
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
    image_tensor = transform_image(image)

    print("Image loaded in {} seconds".format(time() - start))

    classes = {0: "glioma", 1: "meningioma", 2: "no-tumor", 3: "pituitary"}

    start = time()
    with torch.no_grad():
        logits = model(image_tensor)
        ps = torch.exp(logits)
        top_p, top_class = ps.topk(1, dim=1)
        prediction = classes[top_class.item()]

    print("Prediction made in {} seconds".format(time() - start))
    
    counter += 1
    now = str(datetime.now())   
    background_tasks.add_task(add_feature_vec_to_DB, now, image_tensor, int(top_class.item()))

    if counter % 4 == 0:
        background_tasks.add_task(data_drift_func)

    # Convert the image to base64 for returning with response
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={"prediction": prediction, "image": img_base64, "model_name": model_name})


Instrumentator().instrument(app).expose(app)
