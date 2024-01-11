from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
import re
from pydantic import BaseModel
from fastapi import HTTPException


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()
database = {"username": [], "password": []}


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}


@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.post("/login/")
def login(username: str, password: str):
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open("database.csv", "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


# get user name and password from database
@app.get("/login_info/")
def login():
    return database


class EmailDomain(BaseModel):
    email: str
    domain_match: str


@app.get("/text_model/")
def contains_email(data: EmailDomain):
    regex = rf"\b[A-Za-z0-9._%+-]+@{data.domain_match}\.[A-Z|a-z]{{2,}}\b"
    is_valid_email = re.fullmatch(regex, data.email) is not None

    response = {
        "input": data.email,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": is_valid_email,
    }
    return response


from fastapi import UploadFile, File
from typing import Optional
import cv2
from fastapi.responses import FileResponse


@app.post("/cv_model/")
async def cv_model(
    data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28
):
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (h, w))
    cv2.imwrite("image_resize.jpg", res)

    # response = {
    #     "input": data,
    #     "message": HTTPStatus.OK.phrase,
    #     "status-code": HTTPStatus.OK,
    # }
    return FileResponse("image_resize.jpg")
