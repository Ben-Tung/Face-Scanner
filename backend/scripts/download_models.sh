#!/usr/bin/env bash
# Fetches the MediaPipe face landmark model used by app/vision/skin_sampling.py.
# Not committed to git (binary blob) — run this once after cloning, or before
# building the Docker image locally (the Dockerfile does this step itself).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p app/vision/models
curl -sL -o app/vision/models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
echo "Downloaded app/vision/models/face_landmarker.task"
