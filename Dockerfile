FROM python:3.8.20-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    make \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt

COPY . .

ENV PORT=10000

CMD gunicorn --bind 0.0.0.0:$PORT app:app
