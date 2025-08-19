Overview
--------

This app captures HDMI input (e.g., console → capture card), and either:

- overlays a custom crosshair using GStreamer (low CPU), or
- runs person detection (YOLOv11) and draws bounding boxes, optionally with the crosshair too.

You control it from a simple web UI. Choose crosshair color and whether to show the crosshair, detection, or both.


Files
-----
- `app.py`: Flask web UI and process orchestrator
- `launch_overlay.sh`: GStreamer-based crosshair overlay (crosshair-only mode)
- `detector.py`: YOLO detection + drawing path (detection mode and both)
- `requirements.txt`: Python dependencies (except OpenCV on Jetson)


Install (Jetson Orin)
---------------------
1) System packages

```bash
sudo apt update
sudo apt install -y python3-opencv gstreamer1.0-tools gstreamer1.0-plugins-good wmctrl xdotool unclutter
```

2) Python deps

```bash
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

3) Ultralytics model (YOLOv11)

The first run will auto-download `yolov11n.pt`. For better performance, consider exporting to TensorRT later:

```bash
# Optional (advanced): TensorRT engine export (requires proper PyTorch + CUDA/TensorRT env)
yolo export model=yolov11n.pt format=engine
# Then set detector to use the .engine file
```


Run
---
```bash
chmod +x launch_overlay.sh
python3 app.py
```

Open `http://<jetson-ip>:5000/` in your browser.

- Crosshair only: check "Show crosshair", leave detection unchecked → uses GStreamer overlay.
- Detection only: check "Show detection boxes (YOLO)" (crosshair unchecked) → runs detector.
- Both: check both → detector draws boxes and the crosshair directly.

Stop everything with the "Stop All" link in the UI.


Notes
-----
- Keep your monitor on the Jetson input to see the composited result.
- If your capture device is not `/dev/video0`, edit `app.py` (launch_detector) or pass a different device path into `detector.py`.
- For performance on Jetson, prefer GPU/display-accelerated sinks; this POC uses OpenCV for universality. You can replace the display path with `nveglglessink` in a custom pipeline later.
- If the `.pt` model is heavy for your board, use a smaller model (e.g., `yolov11n.pt`) or export to TensorRT.

