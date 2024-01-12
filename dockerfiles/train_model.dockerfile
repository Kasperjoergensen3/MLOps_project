# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install dvc

COPY data.dvc data.dvc

# Initialize DVC
RUN dvc init --no-scm
RUN dvc remote add -d storage gdrive://1OeoRM94MO_zklc4k_k2jDJ4BURYAnkks
# Pull data using DVC
RUN dvc pull --verbose --no-run-cache


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN make requirements

ENTRYPOINT ["python", "-u", "src/train.py"]
