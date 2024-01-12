# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy the Makefile and requirements file
COPY Makefile .
COPY requirements.txt .

# Install Python dependencies
RUN make requirements

# Initialize DVC
COPY data.dvc data.dvc
RUN dvc init --no-scm
RUN dvc remote add -d storage gs://brain_tumor_mlops/
# Pull data using DVC
RUN dvc pull --verbose

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/

WORKDIR /

ENTRYPOINT ["pytest"]
