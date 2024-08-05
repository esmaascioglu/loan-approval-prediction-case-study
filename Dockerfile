FROM python:3.10-slim

RUN apt-get update && apt-get -y install gcc

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY api.py ./api.py
COPY src ./src
COPY models ./models
COPY data ./data

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["python3", "api.py"]
