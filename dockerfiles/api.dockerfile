FROM python:3.10-slim
WORKDIR /code

COPY ./API/ code/API/
COPY ./models/test_predict/ code/models/test_predict/
COPY ./src/ code/src/
COPY pyproject.toml code/pyproject.toml
RUN pip install --no-cache-dir --upgrade -r code/API/app/requirements.txt
RUN pip install -e .

CMD ["uvicorn", "API.app.main:app", "--host", "0.0.0.0", "--port", "80"]