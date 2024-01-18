# API dockerfile

FROM python:3.10-slim

COPY ./API/ code/API/
COPY ./src/ code/src/
COPY pyproject.toml code/pyproject.toml
RUN pip install --no-cache-dir --upgrade -r code/API/app/requirements.txt
RUN pip install -e code/
COPY requirements.txt code/requirements.txt
RUN pip install -r code/requirements.txt


WORKDIR code/
CMD ["uvicorn", "API.app.main:app", "--host", "0.0.0.0", "--port", "80"]