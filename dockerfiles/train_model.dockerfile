# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y dvc
RUN dvc init
RUN dvc pull


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
#COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN pip install -e .

ENTRYPOINT ["python", "-u", "src/train.py"]