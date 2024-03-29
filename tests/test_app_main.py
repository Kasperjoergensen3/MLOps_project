import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import os

models_dir_exists = os.path.isdir("models")
if models_dir_exists:
    from API.app.main import app


@pytest.mark.skipif(not models_dir_exists, reason="Models directory does not exist")
@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.skipif(not models_dir_exists, reason="Models directory does not exist")
@pytest.fixture
def test_image():
    # Create a simple image for testing
    image = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)  # Important: reset buffer to the beginning
    return img_byte_arr


@pytest.mark.skipif(not models_dir_exists, reason="Models directory does not exist")
def test_connection(client):
    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.skipif(not models_dir_exists, reason="Models directory does not exist")
def test_inference_with_vit(client, test_image):
    files = {
        "data": ("test_image.jpg", test_image, "image/jpeg"),
        "model": (None, "ViT"),  # Don't need a path
    }
    response = client.post("/inference/", files=files)
    response_json = response.json()
    assert response.status_code == 200
    assert "prediction" in response_json
    assert "image" in response_json
    assert "model_name" in response_json


@pytest.mark.skipif(not models_dir_exists, reason="Models directory does not exist")
def test_inference_with_simplecnn(client, test_image):
    files = {
        "data": ("test_image.jpg", test_image, "image/jpeg"),
        "model": (None, "SimpleCNN"),
    }
    response = client.post("/inference/", files=files)
    response_json = response.json()
    assert response.status_code == 200
    assert "prediction" in response_json
    assert "image" in response_json
    assert "model_name" in response_json
