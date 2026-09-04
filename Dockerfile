FROM python:3.8.20-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    make \
    wget \
    unzip \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Deno for yt-dlp's YouTube JavaScript challenge handling.
RUN wget -qO- https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
    -O /tmp/deno.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/deno \
    && rm /tmp/deno.zip

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/work /app/static

ENV PORT=10000
ENV PYTHONUNBUFFERED=1

CMD gunicorn \
    --timeout 1800 \
    --workers 1 \
    --bind 0.0.0.0:$PORT \
    app:app
