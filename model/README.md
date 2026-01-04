# Model Artifacts

This repository expects YOLO weights under this `model/` directory. Place your trained checkpoint (e.g., `yolo_rs.pt`) here and restart the Flask service to load it automatically.

- Default name: `yolo_rs.pt` (override with `YOLO_MODEL`).
- Format: Ultralytics YOLO `.pt` weights.
- Notes: If no weights are present, the web app falls back to simulated detections so the UI remains usable without hardware.

Keep large binaries in this folder only and avoid committing private or licensed datasets.
