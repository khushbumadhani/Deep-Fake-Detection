# Deep Fake Detection

## Project Overview

This repository contains a Flask-based deepfake detection web app.
It uses a pre-trained `inception_v3` model and `facenet-pytorch` face extraction to classify uploaded videos as `REAL` or `FAKE`.

Key files:
- `app.py` - Flask application and prediction logic
- `Dockerfile` - Docker image build instructions
- `docker-compose.yml` - Compose service definition for local development
- `requirements.txt` - Python dependencies
- `model.pth` - Pre-trained model weights required by `app.py`
- `templates/index.html` - Frontend upload page

## Setup

### Prerequisites

- Docker Desktop installed and running
- `docker compose` available in your shell
- `model.pth` present in the project root

### Build and run with Docker Compose

From the project root:

```powershell
cd "C:\Users\hp\OneDrive\Desktop\work\Deep-Fake-Detection"
docker compose up -d --build
```

This will:
- build the Docker image
- start the Flask app on port `5000`
- mount the local `videos` and `data` folders into the container

### Open the app

Visit:

```text
http://localhost:5000
```

### Stop the app

```powershell
docker compose down
```

## Alternative: plain Docker

Build the image manually:

```powershell
docker build -t deep-fake-detection .
```

Run the container:

```powershell
docker run --rm -p 5000:5000 --name deep-fake-detection -v ${PWD}:/app -v ${PWD}\videos:/app/videos -v ${PWD}\data:/app/data deep-fake-detection
```

## Notes

- If `docker compose up --build` fails, check the `requirements.txt` package versions.
- The app requires `model.pth` in the repo root and that file must exist before starting the container.
- Uploaded videos are saved under `./videos` and extracted frames are stored under `./data`.
