# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install dvc

COPY data.dvc data.dvc

# Initialize DVC
RUN dvc init --no-scm
RUN dvc remote add -d storage gs://brain_tumor_mlops/
# Pull data using DVC
RUN dvc pull --verbose

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/

WORKDIR /
RUN make requirements

ENTRYPOINT ["pytest"]
