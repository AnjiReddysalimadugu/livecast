#!/usr/bin/env python3
"""Offline smoke test: YuNet loads and detects a drawn face-like blob / blank."""
import sys
from pathlib import Path

import cv2
import numpy as np

model = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
assert model.exists(), f"missing model {model}"

det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.4, 0.3, 5)

# Blank should be 0 faces
blank = np.zeros((480, 640, 3), dtype=np.uint8)
det.setInputSize((640, 480))
_, faces = det.detect(blank)
n_blank = 0 if faces is None else len(faces)

# Synthetic skin-tone oval (weak) — mainly verify detector runs without crash
img = np.full((480, 640, 3), 40, dtype=np.uint8)
cv2.ellipse(img, (320, 220), (70, 90), 0, 0, 360, (180, 160, 140), -1)
cv2.circle(img, (295, 200), 8, (40, 40, 40), -1)
cv2.circle(img, (345, 200), 8, (40, 40, 40), -1)
_, faces2 = det.detect(img)
n_synth = 0 if faces2 is None else len(faces2)

print(f"yunet_ok blank_faces={n_blank} synth_faces={n_synth}")
sys.exit(0)
