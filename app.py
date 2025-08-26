import os
import time
import signal
import subprocess
from typing import Optional, List, Tuple

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

# Runtime process handle
OVERLAY_PROC: Optional[subprocess.Popen] = None


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


def stop_overlay() -> None:
    global OVERLAY_PROC
    try:
        subprocess.run(["pkill", "-f", "cv_display.py"], check=False)
    except Exception:
        pass
    if OVERLAY_PROC is not None:
        try:
            OVERLAY_PROC.terminate()
            t0 = time.time()
            while OVERLAY_PROC.poll() is None and time.time() - t0 < 1.0:
                time.sleep(0.05)
            if OVERLAY_PROC.poll() is None:
                OVERLAY_PROC.kill()
        except Exception:
            try:
                OVERLAY_PROC.kill()
            except Exception:
                pass
    OVERLAY_PROC = None


def hex_to_bgr(color_hex: str) -> str:
    if not color_hex.startswith('#'):
        color_hex = '#' + color_hex
    hh = color_hex.lstrip('#')
    r, g, b = tuple(int(hh[i:i+2], 16) for i in (0, 2, 4))
    return f"{b},{g},{r}"


# Best-model selection (.engine preferred, then smallest variant n>s>m>l>x)
from typing import List

def find_candidate_models(search_dirs: List[str]) -> List[str]:
    candidates: List[str] = []
    exts = (".engine", ".pt")
    for d in search_dirs:
        try:
            for name in os.listdir(d):
                lower = name.lower()
                if lower.endswith(exts) and ("yolo" in lower or "yolov" in lower):
                    candidates.append(os.path.join(d, name))
        except Exception:
            continue
    return candidates


def size_rank_from_name(name: str) -> int:
    lower = name.lower()
    order = ["n", "s", "m", "l", "x"]
    for i, tag in enumerate(order):
        if f"v11{tag}" in lower or f"v{tag}" in lower or lower.endswith(f"{tag}.pt") or lower.endswith(f"{tag}.engine"):
            return i
    return len(order)


def pick_best_model(paths: List[str]) -> str:
    def score(p: str):
        lower = p.lower()
        ext = os.path.splitext(lower)[1]
        ext_priority = 0 if ext == ".engine" else 1
        size_rank = size_rank_from_name(lower)
        return (ext_priority, size_rank, len(lower))
    if not paths:
        return os.path.join(BASE_DIR, "yolov11n.pt")
    return sorted(paths, key=score)[0]


def select_best_ultralytics_model() -> str:
    home = os.path.expanduser("~")
    search_dirs = [
        BASE_DIR,
        os.path.join(BASE_DIR, "models"),
        os.path.join(home, "models"),
        home,
    ]
    return pick_best_model(find_candidate_models(search_dirs))


def launch_overlay(enable_crosshair: bool, enable_detection: bool, crosshair_hex: str, detect_hex: str) -> str:
    global OVERLAY_PROC
    stop_overlay()

    model_path = select_best_ultralytics_model()
    # Tuned defaults for smooth, low-latency feel
    args = [
        "python3", CV_DISPLAY_SCRIPT,
        "--device", DEVICE,
        "--model", model_path,
        "--conf", "0.42",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--fps", str(FPS),
        "--imgsz", "480",
        "--ai-fps", "24",
        "--persist-ms", "220",
        "--max-det", "30",
        "--no-label",
    ]
    if enable_crosshair and os.path.exists(CROSSHAIR_PATH):
        args.extend(["--crosshair", CROSSHAIR_PATH,
                    "--crosshair-color", hex_to_bgr(crosshair_hex)])
    if enable_detection:
        args.extend(["--detection-color", hex_to_bgr(detect_hex)])
    else:
        args.append("--no-detect")

    OVERLAY_PROC = subprocess.Popen(args)
    return model_path


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
            model_used = launch_overlay(enable_crosshair, enable_detection, crosshair_color, detection_color)
            base_msg = f"✅ Using model: {os.path.basename(model_used)}"
            if enable_crosshair and enable_detection:
                status_message = base_msg + " | Crosshair + AI detection launched"
            elif enable_crosshair:
                status_message = base_msg + " | Crosshair overlay launched"
            elif enable_detection:
                status_message = base_msg + " | AI detection launched"
            else:
                status_message = "✅ All overlays stopped"
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
def stop_all():
    stop_overlay()
    return redirect(url_for("index", msg="🛑 Stopped", cls="success"))


if __name__ == "__main__":
    try:
        print("🎮 Gaming Overlay Control starting...")
        print("📱 Web UI: http://localhost:5000")
        print("💡 Best perf: place yolov11n.engine in project root or ./models/")
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    finally:
        stop_overlay()