# docker build -f dockerfiles/api.dockerfile . -t api:latest   
# docker run -d --name mycontainer -p 80:80 api:latest       

FROM python:3.10-slim

COPY ./API/ code/API/
COPY ./models/test_predict/ code/models/test_predict/
COPY ./src/ code/src/
COPY pyproject.toml code/pyproject.toml
RUN pip install --no-cache-dir --upgrade -r code/API/app/requirements.txt
RUN pip install -e code/

WORKDIR code/
CMD ["uvicorn", "API.app.main:app", "--host", "0.0.0.0", "--port", "80"]