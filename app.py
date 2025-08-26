import os
import time
import glob
import signal
import subprocess
from typing import Optional

from flask import Flask, render_template_string, send_file, request, redirect, url_for, abort
from PIL import Image


app = Flask(__name__)

# Use current working directory instead of hardcoded path
BASE_DIR = os.getcwd()
CROSSHAIR_PATH = os.path.join(BASE_DIR, "crosshair.png")
SCRIPT_PATH = os.path.join(BASE_DIR, "launch_overlay.sh")
PREVIEW_FOLDER = os.path.join(BASE_DIR, "static")

# Runtime process handles
DETECTOR_PROC: Optional[subprocess.Popen] = None
GST_PROC: Optional[subprocess.Popen] = None


HTML = """
<!doctype html>
<title>Overlay Control</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
.container { max-width: 600px; }
.form-group { margin: 15px 0; }
button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
button:hover { background: #0056b3; }
.stop-link { color: #dc3545; text-decoration: none; margin-left: 10px; }
.stop-link:hover { text-decoration: underline; }
.preview { margin: 20px 0; }
.status { margin: 10px 0; padding: 10px; border-radius: 4px; }
.status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
.status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
</style>
<div class="container">
<h2>Overlay Control</h2>
<form method="post">
  <div class="form-group">
    <label><input type="checkbox" name="enable_crosshair" %ENABLE_CROSSHAIR%> Show crosshair</label>
  </div>
  <div class="form-group">
    <label><input type="checkbox" name="enable_detection" %ENABLE_DETECTION%> Show detection boxes (YOLO)</label>
  </div>
  <div class="form-group">
    <button type="submit">Apply & Launch</button>
    <a href="/stop" class="stop-link">Stop All</a>
  </div>
</form>

<div class="preview">
  <p>Current Crosshair (scaled preview):</p>
  <img src="/preview" style="border:1px solid #000; width:100px;">
</div>

<div class="status %STATUS_CLASS%">
  %STATUS_MESSAGE%
</div>
</div>
"""


def ensure_directories() -> None:
    """Ensure required directories exist."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)


def create_default_crosshair() -> None:
    """Create a default crosshair if none exists."""
    if os.path.exists(CROSSHAIR_PATH):
        return
    
    try:
        from PIL import Image, ImageDraw
        
        # Create a 64x64 transparent image
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple crosshair
        center = 32
        color = (0, 255, 0, 255)  # Green
        thickness = 3
        
        # Horizontal line
        draw.line([(8, center), (24, center)], fill=color, width=thickness)
        draw.line([(40, center), (56, center)], fill=color, width=thickness)
        
        # Vertical line
        draw.line([(center, 8), (center, 24)], fill=color, width=thickness)
        draw.line([(center, 40), (center, 56)], fill=color, width=thickness)
        
        img.save(CROSSHAIR_PATH)
        print(f"Created default crosshair at {CROSSHAIR_PATH}")
    except Exception as e:
        print(f"Failed to create default crosshair: {e}")


def crosshair_size() -> tuple[int, int]:
    """Get crosshair image dimensions."""
    if not os.path.exists(CROSSHAIR_PATH):
        return (0, 0)
    try:
        with Image.open(CROSSHAIR_PATH) as im:
            return im.size
    except Exception:
        return (0, 0)


def cleanup_old_crosshairs() -> None:
    """Clean up old timestamped crosshair files."""
    pattern = os.path.join(BASE_DIR, "crosshair_*.png")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except Exception:
            pass


def stop_gst() -> None:
    """Stop GStreamer processes."""
    global GST_PROC
    try:
        subprocess.run(["pkill", "-f", "gst-launch-1.0"], check=False)
    except Exception:
        pass
    if GST_PROC is not None:
        try:
            GST_PROC.terminate()
            GST_PROC.wait(timeout=3)
        except (subprocess.TimeoutExpired, Exception):
            try:
                GST_PROC.kill()
            except Exception:
                pass
        GST_PROC = None


def stop_detector() -> None:
    """Stop detector processes."""
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
    """Launch GStreamer overlay with crosshair."""
    global GST_PROC
    stop_gst()
    
    if not os.path.isfile(SCRIPT_PATH):
        raise FileNotFoundError(f"launch script not found: {SCRIPT_PATH}")
    
    if not os.path.isfile(crosshair_path):
        raise FileNotFoundError(f"crosshair image not found: {crosshair_path}")
    
    GST_PROC = subprocess.Popen([
        "/bin/bash", SCRIPT_PATH, crosshair_path, str(offset_x), str(offset_y)
    ])


def launch_detector(enable_crosshair: bool, crosshair_path: str) -> None:
    """Launch detector with optional crosshair."""
    global DETECTOR_PROC
    stop_detector()
    
    detector_script = os.path.join(BASE_DIR, "detector.py")
    if not os.path.isfile(detector_script):
        raise FileNotFoundError(f"detector script not found: {detector_script}")
    
    args = [
        "python3", detector_script,
        "--device", "/dev/video0",
        "--model", "yolov11n.pt",  # Fixed model name
        "--conf", "0.4",
        "--fullscreen", "1",
        "--window-title", "Detector",
        "--display", "1",
    ]
    
    if enable_crosshair and crosshair_path:
        args.extend(["--crosshair-enable", "1", "--crosshair-image", crosshair_path])
    else:
        args.extend(["--crosshair-enable", "0"])
    
    DETECTOR_PROC = subprocess.Popen(args)


@app.before_request
def init_app():
    """Initialize application on first request."""
    ensure_directories()
    create_default_crosshair()


@app.route("/", methods=["GET", "POST"])
def index():
    """Main page with form handling."""
    status_message = ""
    status_class = "success"
    
    if request.method == "POST":
        enable_crosshair = request.form.get("enable_crosshair") == "on"
        enable_detection = request.form.get("enable_detection") == "on"

        if not os.path.exists(CROSSHAIR_PATH):
            status_message = "Error: crosshair.png not found"
            status_class = "error"
        else:
            try:
                # Compute centered offsets for 1920x1080
                cw, ch = crosshair_size()
                video_w, video_h = 1920, 1080
                offset_x = max(0, (video_w - cw) // 2)
                offset_y = max(0, (video_h - ch) // 2)
                cleanup_old_crosshairs()

                # Stop existing processes
                stop_detector()
                stop_gst()

                if enable_detection:
                    # Use detector with optional crosshair
                    launch_detector(enable_crosshair, CROSSHAIR_PATH)
                    status_message = "Detection started successfully"
                elif enable_crosshair:
                    # Use GStreamer overlay for crosshair only
                    launch_gst_overlay(CROSSHAIR_PATH, offset_x, offset_y)
                    status_message = "Crosshair overlay started successfully"
                else:
                    status_message = "All overlays stopped"
                    
            except Exception as e:
                status_message = f"Error starting overlay: {str(e)}"
                status_class = "error"

        return redirect(url_for("index", ec=int(enable_crosshair), ed=int(enable_detection), 
                               msg=status_message, cls=status_class))

    # GET request
    ec = request.args.get("ec", default="0")
    ed = request.args.get("ed", default="0")
    status_message = request.args.get("msg", default="")
    status_class = request.args.get("cls", default="success")
    
    html = HTML.replace("%ENABLE_CROSSHAIR%", "checked" if ec == "1" else "")
    html = html.replace("%ENABLE_DETECTION%", "checked" if ed == "1" else "")
    html = html.replace("%STATUS_MESSAGE%", status_message)
    html = html.replace("%STATUS_CLASS%", status_class)
    
    return render_template_string(html)


@app.route("/preview")
def preview():
    """Serve crosshair preview image."""
    if not os.path.exists(CROSSHAIR_PATH):
        abort(404)
    return send_file(CROSSHAIR_PATH, mimetype="image/png")


@app.route("/stop")
def stop_all():
    """Stop all running processes."""
    stop_detector()
    stop_gst()
    return redirect(url_for("index", msg="All processes stopped", cls="success"))


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    finally:
        stop_detector()
        stop_gst()