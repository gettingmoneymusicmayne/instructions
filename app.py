import os
import time
import signal
import subprocess
from typing import Optional

from flask import Flask, render_template_string, request, redirect, url_for


app = Flask(__name__)

# Config
WIDTH = int(os.getenv("OVERLAY_WIDTH", "1920"))
HEIGHT = int(os.getenv("OVERLAY_HEIGHT", "1080"))
FPS = int(os.getenv("OVERLAY_FPS", "60"))
DEVICE = os.getenv("OVERLAY_DEVICE", "/dev/video0")

# Use current working directory
BASE_DIR = os.getcwd()
CROSSHAIR_PATH = os.path.join(BASE_DIR, "crosshair.png")
LAUNCH_SCRIPT = os.path.join(BASE_DIR, "launch_overlay.sh")
CV_DISPLAY_SCRIPT = os.path.join(BASE_DIR, "cv_display.py")

# Runtime process handles
DISPLAY_PROC: Optional[subprocess.Popen] = None
PUBLISHER_PROC: Optional[subprocess.Popen] = None


HTML = """
<!doctype html>
<html>
<head>
    <title>Gaming Overlay Control</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            min-height: 100vh;
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        h1 { 
            text-align: center; 
            margin-bottom: 30px;
            color: #fff;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .form-group { 
            margin: 20px 0; 
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        .form-group label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #fff;
        }
        .checkbox-group { display: flex; gap: 20px; flex-wrap: wrap; }
        .checkbox-item { display: flex; align-items: center; gap: 8px; }
        .checkbox-item input[type="checkbox"] { transform: scale(1.2); }
        .color-group { display: flex; align-items: center; gap: 15px; margin-top: 10px; }
        .color-group label { margin: 0; min-width: 80px; }
        .color-group input[type="color"] { width: 50px; height: 40px; border: none; border-radius: 5px; cursor: pointer; }
        .button-group { display: flex; gap: 15px; margin-top: 30px; }
        button { padding: 12px 24px; background: linear-gradient(45deg, #4CAF50, #45a049); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; transition: all 0.3s ease; flex: 1; }
        button:hover { background: linear-gradient(45deg, #45a049, #4CAF50); transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); }
        .stop-btn { background: linear-gradient(45deg, #f44336, #d32f2f) !important; }
        .stop-btn:hover { background: linear-gradient(45deg, #d32f2f, #f44336) !important; }
        .status { margin: 20px 0; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; }
        .status.success { background: rgba(76, 175, 80, 0.3); color: #4CAF50; border: 1px solid #4CAF50; }
        .status.error { background: rgba(244, 67, 54, 0.3); color: #f44336; border: 1px solid #f44336; }
        .info { background: rgba(33, 150, 243, 0.3); color: #2196F3; border: 1px solid #2196F3; padding: 15px; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Gaming Overlay Control</h1>
        
        <form method="post">
            <div class="form-group">
                <label>Overlay Features:</label>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="checkbox" name="enable_crosshair" %ENABLE_CROSSHAIR% id="crosshair">
                        <label for="crosshair">Custom Crosshair</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" name="enable_detection" %ENABLE_DETECTION% id="detection">
                        <label for="detection">AI Person Detection</label>
                    </div>
                </div>
            </div>

            <div class="form-group">
                <label>Colors:</label>
                <div class="color-group">
                    <label for="crosshair_color">Crosshair:</label>
                    <input type="color" name="crosshair_color" id="crosshair_color" value="%CROSSHAIR_COLOR%">
                </div>
                <div class="color-group">
                    <label for="detection_color">AI Boxes:</label>
                    <input type="color" name="detection_color" id="detection_color" value="%DETECTION_COLOR%">
                </div>
            </div>

            <div class="button-group">
                <button type="submit">Apply & Launch</button>
                <a href="/stop" class="stop-btn" style="text-decoration: none; display: flex; align-items: center; justify-content: center;">Stop All</a>
            </div>
        </form>

        <div class="status %STATUS_CLASS%">
            %STATUS_MESSAGE%
        </div>

        <div class="info">
            <strong>Setup:</strong> Capture card input → Console/PC, Output → Monitor, USB → Jetson<br>
            <strong>Display:</strong> Monitor shows original video with overlays via DisplayPort
        </div>
    </div>
</body>
</html>
"""


def stop_all_procs() -> None:
    global DISPLAY_PROC, PUBLISHER_PROC
    try:
        subprocess.run(["pkill", "-f", "cv_display.py"], check=False)
        subprocess.run(["pkill", "-f", "gst-launch-1.0"], check=False)
        subprocess.run(["pkill", "-f", "launch_overlay.sh"], check=False)
    except Exception:
        pass
    for proc in (PUBLISHER_PROC, DISPLAY_PROC):
        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    DISPLAY_PROC = None
    PUBLISHER_PROC = None


def hex_to_bgr(color_hex: str) -> str:
    if not color_hex.startswith('#'):
        color_hex = '#' + color_hex
    hh = color_hex.lstrip('#')
    r, g, b = tuple(int(hh[i:i+2], 16) for i in (0, 2, 4))
    return f"{b},{g},{r}"


def launch_pipeline(enable_crosshair: bool, enable_detection: bool, crosshair_hex: str, detect_hex: str) -> None:
    global DISPLAY_PROC, PUBLISHER_PROC
    stop_all_procs()

    # Start display pipeline (60fps compositor)
    DISPLAY_PROC = subprocess.Popen(["/bin/bash", LAUNCH_SCRIPT, DEVICE, str(WIDTH), str(HEIGHT), str(FPS)])
    time.sleep(1.0)

    # Start overlay publisher (our cv_display.py)
    args = [
        "python3", CV_DISPLAY_SCRIPT,
        "--width", str(WIDTH), "--height", str(HEIGHT), "--fps", str(FPS),
        "--imgsz", "480", "--skip", "2", "--conf", "0.35",
        "--detection-color", hex_to_bgr(detect_hex),
    ]
    if enable_crosshair and os.path.exists(CROSSHAIR_PATH):
        args.extend(["--crosshair", CROSSHAIR_PATH, "--crosshair-color", hex_to_bgr(crosshair_hex)])
    if not enable_detection:
        args.append("--no-detect")
    PUBLISHER_PROC = subprocess.Popen(args)


@app.route("/", methods=["GET", "POST"])
def index():
    status_message = ""
    status_class = "success"

    if request.method == "POST":
        enable_crosshair = request.form.get("enable_crosshair") == "on"
        enable_detection = request.form.get("enable_detection") == "on"
        crosshair_color = request.form.get("crosshair_color", "#00ff00")
        detection_color = request.form.get("detection_color", "#ffff00")

        try:
            launch_pipeline(enable_crosshair, enable_detection, crosshair_color, detection_color)
            if enable_crosshair and enable_detection:
                status_message = "✅ Crosshair + AI detection running at 60fps render"
            elif enable_crosshair:
                status_message = "✅ Crosshair overlay running at 60fps render"
            elif enable_detection:
                status_message = "✅ AI detection running at 60fps render"
            else:
                status_message = "✅ All stopped"
        except Exception as e:
            status_message = f"❌ Error: {e}"
            status_class = "error"

        return redirect(url_for("index",
                               ec=int(enable_crosshair), ed=int(enable_detection),
                               cc=crosshair_color, dc=detection_color,
                               msg=status_message, cls=status_class))

    ec = request.args.get("ec", default="0")
    ed = request.args.get("ed", default="0")
    cc = request.args.get("cc", default="#00ff00")
    dc = request.args.get("dc", default="#ffff00")
    status_message = request.args.get("msg", default="Ready")
    status_class = request.args.get("cls", default="success")

    html = HTML.replace("%ENABLE_CROSSHAIR%", "checked" if ec == "1" else "")
    html = html.replace("%ENABLE_DETECTION%", "checked" if ed == "1" else "")
    html = html.replace("%CROSSHAIR_COLOR%", cc)
    html = html.replace("%DETECTION_COLOR%", dc)
    html = html.replace("%STATUS_MESSAGE%", status_message)
    html = html.replace("%STATUS_CLASS%", status_class)
    return render_template_string(html)


@app.route("/stop")
def stop_route():
    stop_all_procs()
    return redirect(url_for("index", msg="🛑 Stopped", cls="success"))


if __name__ == "__main__":
    try:
        print("🎮 Gaming Overlay Control starting...")
        print("📱 Web UI: http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    finally:
        stop_all_procs()