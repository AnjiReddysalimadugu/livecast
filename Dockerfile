FROM golang:1.24-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /livecast .

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /livecast /app/livecast
COPY web /app/web
COPY whisper /app/whisper
COPY facedetect /app/facedetect
COPY translate /app/translate
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"

ENV RENDER=true \
    LIVECAST_CLOUD=1 \
    WHISPER_MODEL=tiny \
    WHISPER_LANGUAGE=en \
    PYTHONUNBUFFERED=1

EXPOSE 10000
# Bind $PORT from Render (default 10000). Shell form so env expands.
CMD ["sh", "-c", "exec /app/livecast -listen 0.0.0.0 -port ${PORT:-10000}"]
