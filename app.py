import os
import time
import glob
import signal
import subprocess
from typing import Optional

from flask import Flask, render_template_string, send_file, request, redirect, url_for, abort
from PIL import Image


app = Flask(__name__)

# Project directory based on this file's location
BASE_DIR = os.path.dirname(__file__)
# Paths
CROSSHAIR_PATH = os.path.join(BASE_DIR, "crosshair.png")
SCRIPT_PATH = os.path.join(BASE_DIR, "launch_overlay.sh")
DISPLAY_PATH = os.path.join(BASE_DIR, "cv_display.py")
PREVIEW_FOLDER = os.path.join(BASE_DIR, "static")

# Capture device
DEVICE_PATH = os.environ.get("CAPTURE_DEVICE", "/dev/video0")

# Runtime process handles
DISPLAY_PROC: Optional[subprocess.Popen] = None

HTML = """
<!doctype html>
<title>Overlay Control</title>
<h2>Overlay Control</h2>
<form method="post">
  <div style="margin-top: 10px;">
    <label><input type="checkbox" name="enable_crosshair" %ENABLE_CROSSHAIR%> Show crosshair</label>
  </div>
  <div>
    <label><input type="checkbox" name="enable_detection" %ENABLE_DETECTION%> Show detection boxes (YOLO)</label>
  </div>
  <div style="margin-top: 10px;">
    <button type="submit">Apply & Launch</button>
    <a href="/stop" style="margin-left: 10px;">Stop All</a>
  </div>
</form>

<p style="margin-top: 20px;">Current Crosshair (scaled preview):</p>
<img src="/preview" style="margin-top:10px; border:1px solid #000; width:100px;">
"""


def ensure_directories() -> None:
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)


def crosshair_size() -> tuple[int, int]:
    if not os.path.exists(CROSSHAIR_PATH):
        return (0, 0)
    with Image.open(CROSSHAIR_PATH) as im:
        return im.size


def stop_display() -> None:
    global DISPLAY_PROC
    if DISPLAY_PROC is not None:
        try:
            DISPLAY_PROC.send_signal(signal.SIGTERM)
            try:
                DISPLAY_PROC.wait(timeout=3)
            except subprocess.TimeoutExpired:
                DISPLAY_PROC.kill()
        except Exception:
            pass
        DISPLAY_PROC = None


def launch_crosshair_only(crosshair_path: str, offset_x: int, offset_y: int) -> None:
    global DISPLAY_PROC
    stop_display()
    if not os.path.isfile(SCRIPT_PATH):
        raise FileNotFoundError(f"launch script not found: {SCRIPT_PATH}")
    DISPLAY_PROC = subprocess.Popen([
        "/bin/bash", SCRIPT_PATH, crosshair_path, str(offset_x), str(offset_y)
    ])


def launch_both_single_window(device_path: str, crosshair_path: str) -> None:
    global DISPLAY_PROC
    stop_display()
    if not os.path.isfile(DISPLAY_PATH):
        raise FileNotFoundError(f"display script not found: {DISPLAY_PATH}")
    DISPLAY_PROC = subprocess.Popen([
        "python3", DISPLAY_PATH,
        "--device", device_path,
        "--width", "1920", "--height", "1080", "--fps", "60",
        "--model", "yolo11n.pt",
        "--conf", "0.4",
        "--crosshair", crosshair_path,
        "--imgsz", "512",
        "--compute-device", "cpu",
        "--half", "0",
        "--display_backend", "gstreamer",
        "--gst_sink", "glimagesink",
        "--gst_sync", "0",
    ])


@app.before_request
def init_app():
    ensure_directories()
    if not os.path.exists(CROSSHAIR_PATH):
        return (
            "<h1>Error:</h1>"
            "<p>Crosshair image not found. Please add <code>crosshair.png</code> in this directory.</p>",
            500,
        )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        enable_crosshair = request.form.get("enable_crosshair") == "on"
        enable_detection = request.form.get("enable_detection") == "on"

        if not os.path.exists(CROSSHAIR_PATH):
            abort(400, "crosshair.png not found")

        # Compute centered offsets for 1920x1080
        cw, ch = crosshair_size()
        video_w, video_h = 1920, 1080
        offset_x = max(0, (video_w - cw) // 2)
        offset_y = max(0, (video_h - ch) // 2)

        if enable_detection:
            launch_both_single_window(DEVICE_PATH, CROSSHAIR_PATH)
        else:
            if enable_crosshair:
                launch_crosshair_only(CROSSHAIR_PATH, offset_x, offset_y)
            else:
                stop_display()

        return redirect(url_for("index", ec=int(enable_crosshair), ed=int(enable_detection)))

    ec = request.args.get("ec", default="1")
    ed = request.args.get("ed", default="0")
    html = HTML.replace("%ENABLE_CROSSHAIR%", "checked" if ec == "1" else "")
    html = html.replace("%ENABLE_DETECTION%", "checked" if ed == "1" else "")
    return render_template_string(html)


@app.route("/preview")
def preview():
    if not os.path.exists(CROSSHAIR_PATH):
        abort(404)
    return send_file(CROSSHAIR_PATH, mimetype="image/png")


@app.route("/stop")
def stop_all():
    stop_display()
    return redirect(url_for("index"))


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    finally:
        stop_display()