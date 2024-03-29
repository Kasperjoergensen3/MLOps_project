# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/processed/test.pt data/processed/test.pt
COPY models/ models/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN pip install -e .

#ENTRYPOINT ["python", "-u", "src/predict_model.py", "--output_dir=models/test_predict", "--checkpoint=best-checkpoint.ckpt"]
ENTRYPOINT ["python", "-u", "src/predict_model.py"]