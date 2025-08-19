import os
import time
import glob
import signal
import subprocess
from typing import Optional

from flask import Flask, render_template_string, send_file, request, redirect, url_for
from PIL import Image, ImageDraw


app = Flask(__name__)

# Fixed project directory as requested
BASE_DIR = "/home/opulentpro/Documents/crosshair-dashboard"
BASE_PATH = os.path.join(BASE_DIR, "crosshair_base.png")
SCRIPT_PATH = os.path.join(BASE_DIR, "launch_overlay.sh")
PREVIEW_FOLDER = os.path.join(BASE_DIR, "static")
PREVIEW_PATH = os.path.join(PREVIEW_FOLDER, "preview.png")


# Runtime process handles
DETECTOR_PROC: Optional[subprocess.Popen] = None
GST_PROC: Optional[subprocess.Popen] = None


HTML = """
<!doctype html>
<title>Overlay Control</title>
<h2>Overlay Control</h2>
<form method="post">
  <div>
    <label>Crosshair color:</label>
    <input type="color" name="color" value="#00ff00">
  </div>
  <div style="margin-top: 10px;">
    <label><input type="checkbox" name="enable_crosshair" checked> Show crosshair</label>
  </div>
  <div>
    <label><input type="checkbox" name="enable_detection"> Show detection boxes (YOLO)</label>
  </div>
  <div style="margin-top: 10px;">
    <button type="submit">Apply & Launch</button>
    <a href="/stop" style="margin-left: 10px;">Stop All</a>
  </div>
</form>

<p style="margin-top: 20px;">Current Crosshair Preview:</p>
<img src="/preview" style="margin-top:10px; border:1px solid #000; width:100px;">
"""


def ensure_directories() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)


def generate_default_base_if_missing() -> None:
    if os.path.exists(BASE_PATH):
        return
    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    center = size // 2
    color = (255, 255, 255, 255)
    thickness = 2
    gap = 10
    line_len = size // 2 - 8
    # Horizontal
    draw.rectangle([center - line_len, center - thickness, center - gap, center + thickness], fill=color)
    draw.rectangle([center + gap, center - thickness, center + line_len, center + thickness], fill=color)
    # Vertical
    draw.rectangle([center - thickness, center - line_len, center + thickness, center - gap], fill=color)
    draw.rectangle([center - thickness, center + gap, center + thickness, center + line_len], fill=color)
    image.save(BASE_PATH, format="PNG")


def cleanup_old_crosshairs(keep_path: str) -> None:
    pattern = os.path.join(BASE_DIR, "crosshair_*.png")
    for path in glob.glob(pattern):
        if os.path.abspath(path) == os.path.abspath(keep_path):
            continue
        try:
            os.remove(path)
        except Exception as ex:
            print(f"Failed to delete {path}: {ex}")


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


def launch_gst_overlay(crosshair_path: str) -> None:
    global GST_PROC
    stop_gst()
    if not os.path.isfile(SCRIPT_PATH):
        raise FileNotFoundError(f"launch script not found: {SCRIPT_PATH}")
    GST_PROC = subprocess.Popen(["/bin/bash", SCRIPT_PATH, crosshair_path])


def launch_detector(enable_crosshair: bool, color_hex: str) -> None:
    global DETECTOR_PROC
    stop_detector()
    args = [
        "python3", os.path.join(BASE_DIR, "detector.py"),
        "--device", "/dev/video0",
        "--model", "yolov11n.pt",
        "--conf", "0.4",
        "--fullscreen", "1",
        "--window-title", "Jetson Overlay",
    ]
    if enable_crosshair:
        args.extend(["--crosshair-enable", "1", "--crosshair-color", color_hex])
    else:
        args.extend(["--crosshair-enable", "0"])
    DETECTOR_PROC = subprocess.Popen(args)


@app.before_request
def init_app():
    ensure_directories()
    generate_default_base_if_missing()
    if not os.path.exists(BASE_PATH):
        return (
            "<h1>Error:</h1>"
            "<p>Base crosshair image not found and could not be created. "
            "Please add <code>crosshair_base.png</code> in the app directory.</p>",
            500,
        )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        hex_color = request.form.get("color", "#00ff00").lstrip("#")
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        enable_crosshair = request.form.get("enable_crosshair") == "on"
        enable_detection = request.form.get("enable_detection") == "on"

        # Build colored crosshair from base for preview and gst overlay mode
        base = Image.open(BASE_PATH).convert("RGBA")
        _, _, _, alpha = base.split()
        color_img = Image.new("RGBA", base.size, rgb + (0,))
        colored = Image.composite(color_img, Image.new("RGBA", base.size, (0, 0, 0, 0)), alpha)
        colored.putalpha(alpha)

        # Save preview and unique PNG for gst overlay
        colored.resize((100, 100)).save(PREVIEW_PATH)
        timestamp = int(time.time() * 1000)
        unique_crosshair = os.path.join(BASE_DIR, f"crosshair_{timestamp}.png")
        colored.save(unique_crosshair, format="PNG", optimize=False)
        cleanup_old_crosshairs(unique_crosshair)

        # Orchestrate processes based on selection
        if enable_detection:
            # Use detector pipeline, optionally drawing crosshair directly
            stop_gst()
            launch_detector(enable_crosshair=enable_crosshair, color_hex=f"#{hex_color}")
        else:
            # Use gst overlay if only crosshair is desired
            stop_detector()
            if enable_crosshair:
                launch_gst_overlay(unique_crosshair)
            else:
                stop_gst()

        return redirect(url_for("index"))

    return render_template_string(HTML)


@app.route("/preview")
def preview():
    if os.path.exists(PREVIEW_PATH):
        return send_file(PREVIEW_PATH, mimetype="image/png")
    else:
        return send_file(BASE_PATH, mimetype="image/png")


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

