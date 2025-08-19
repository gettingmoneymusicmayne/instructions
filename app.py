import os
import time
import glob
import signal
import subprocess
from typing import Optional

from flask import Flask, render_template_string, send_file, request, redirect, url_for, abort
from PIL import Image


app = Flask(__name__)

# Fixed project directory as requested
BASE_DIR = "/home/opulentpro/Documents/crosshair-dashboard"
# User-provided crosshair image (no auto-generation)
CROSSHAIR_PATH = os.path.join(BASE_DIR, "crosshair.png")
SCRIPT_PATH = os.path.join(BASE_DIR, "launch_overlay.sh")
PREVIEW_FOLDER = os.path.join(BASE_DIR, "static")


# Runtime process handles
DETECTOR_PROC: Optional[subprocess.Popen] = None
GST_PROC: Optional[subprocess.Popen] = None


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
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)


def crosshair_size() -> tuple[int, int]:
    if not os.path.exists(CROSSHAIR_PATH):
        return (0, 0)
    with Image.open(CROSSHAIR_PATH) as im:
        return im.size


def cleanup_old_crosshairs() -> None:
    # No-op now; we no longer generate timestamped crosshairs
    pattern = os.path.join(BASE_DIR, "crosshair_*.png")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except Exception:
            pass


def stop_gst() -> None:
    global GST_PROC
    try:
        subprocess.run(["pkill", "-f", "gst-launch-1.0"], check=False)
    except Exception:
        pass
    if GST_PROC is not None:
        try:
            GST_PROC.terminate()
        except Exception:
            pass
        GST_PROC = None


def stop_detector() -> None:
    global DETECTOR_PROC
    if DETECTOR_PROC is not None:
        try:
            DETECTOR_PROC.send_signal(signal.SIGTERM)
            try:
                DETECTOR_PROC.wait(timeout=3)
            except subprocess.TimeoutExpired:
                DETECTOR_PROC.kill()
        except Exception:
            pass
        DETECTOR_PROC = None


def launch_gst_overlay(crosshair_path: str, offset_x: int, offset_y: int) -> None:
    global GST_PROC
    stop_gst()
    if not os.path.isfile(SCRIPT_PATH):
        raise FileNotFoundError(f"launch script not found: {SCRIPT_PATH}")
    GST_PROC = subprocess.Popen(["/bin/bash", SCRIPT_PATH, crosshair_path, str(offset_x), str(offset_y)])


def launch_detector(display_output: bool) -> None:
    global DETECTOR_PROC
    stop_detector()
    args = [
        "python3", os.path.join(BASE_DIR, "detector.py"),
        "--device", "/dev/video0",
        "--model", "yolo11n.pt",
        "--conf", "0.4",
        "--fullscreen", "1" if display_output else "0",
        "--window-title", "Detector" if display_output else "",
        "--display", "1" if display_output else "0",
    ]
    DETECTOR_PROC = subprocess.Popen(args)


@app.before_request
def init_app():
    ensure_directories()
    if not os.path.exists(CROSSHAIR_PATH):
        return (
            "<h1>Error:</h1>"
            "<p>Crosshair image not found. Please add <code>crosshair.png</code> in /home/opulentpro/Documents/crosshair-dashboard.</p>",
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
        cleanup_old_crosshairs()

        # Orchestrate processes based on selection
        if enable_detection:
            # Use shm-decoupled display + detector for robust caps and low latency
            stop_gst()
            stop_detector()
            DISPLAY_CMD = [
                "/bin/bash", os.path.join(BASE_DIR, "display_both.sh"),
                "/dev/video0", "1920", "1080", "60"
            ]
            DETECTOR_CMD = [
                "python3", os.path.join(BASE_DIR, "detector_shm.py"),
                "--model", "yolo11n.pt",
                "--conf", "0.4",
                "--width", "1920", "--height", "1080", "--fps", "60",
            ]
            if enable_crosshair:
                DETECTOR_CMD.extend(["--crosshair", CROSSHAIR_PATH])
            # Start display first so shmsink is listening
            global GST_PROC, DETECTOR_PROC
            GST_PROC = subprocess.Popen(DISPLAY_CMD)
            time.sleep(0.5)
            DETECTOR_PROC = subprocess.Popen(DETECTOR_CMD)
        else:
            # Use gst overlay if only crosshair is desired
            stop_detector()
            if enable_crosshair:
                launch_gst_overlay(CROSSHAIR_PATH, offset_x, offset_y)
            else:
                stop_gst()

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
    stop_detector()
    stop_gst()
    return redirect(url_for("index"))


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    finally:
        stop_detector()
        stop_gst()

