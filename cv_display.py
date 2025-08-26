#!/usr/bin/env python3
"""
Gaming Overlay System - Overlay Publisher
Reads BGR frames from /tmp/capture_bgr (GStreamer shmsrc) and publishes BGRA overlay frames to /tmp/overlay_rgba.
Performs YOLO person detection on downscaled copies for performance and draws boxes + crosshair onto overlay.
"""

import argparse
import signal
import time
import sys
import os

import cv2
import numpy as np


CAPTURE_SOCK = "/tmp/capture_bgr"
OVERLAY_SOCK = "/tmp/overlay_rgba"


def parse_args():
    parser = argparse.ArgumentParser("Gaming Overlay - Overlay Publisher")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS for overlay")
    parser.add_argument("--model", default="yolov11n.pt", help="YOLO model path or .engine")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence")
    parser.add_argument("--imgsz", type=int, default=480, help="Inference size")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections per image")
    parser.add_argument("--skip", type=int, default=2, help="Run detection every N frames (>=1)")
    parser.add_argument("--crosshair", default="", help="Crosshair image path")
    parser.add_argument("--crosshair-color", default="0,255,0", help="Crosshair color (BGR)")
    parser.add_argument("--detection-color", default="0,255,255", help="Detection box color (BGR, alpha 220)")
    parser.add_argument("--crosshair-scale", type=float, default=1.0, help="Crosshair scale")
    parser.add_argument("--no-label", action="store_true", help="Do not draw confidence labels")
    parser.add_argument("--no-detect", action="store_true", help="Disable YOLO detection (crosshair only)")
    return parser.parse_args()


def parse_bgr_color(color_str: str) -> tuple:
    try:
        b, g, r = map(int, color_str.split(','))
        return (b, g, r)
    except Exception:
        return (0, 255, 0)


def load_crosshair_bgra(path: str, scale: float = 1.0, color: tuple = (0, 255, 0)):
    if not path or not os.path.exists(path):
        return None
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        if color != (0, 255, 0):
            alpha = img[:, :, 3]
            mask = alpha > 0
            img[mask, 0] = color[0]
            img[mask, 1] = color[1]
            img[mask, 2] = color[2]
        if scale != 1.0:
            new_w = max(1, int(img.shape[1] * scale))
            new_h = max(1, int(img.shape[0] * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        print(f"Error loading crosshair: {e}", file=sys.stderr)
        return None


def overlay_crosshair_bgra(dst_rgba: np.ndarray, crosshair_bgra: np.ndarray) -> None:
    if crosshair_bgra is None:
        return
    fh, fw = dst_rgba.shape[:2]
    ch, cw = crosshair_bgra.shape[:2]
    x = max(0, (fw - cw) // 2)
    y = max(0, (fh - ch) // 2)
    x2 = min(fw, x + cw)
    y2 = min(fh, y + ch)
    cw_eff = x2 - x
    ch_eff = y2 - y
    if cw_eff <= 0 or ch_eff <= 0:
        return
    roi = dst_rgba[y:y2, x:x2]
    ch_crop = crosshair_bgra[0:ch_eff, 0:cw_eff]
    if ch_crop.shape[2] == 4:
        overlay_rgb = ch_crop[:, :, :3].astype(np.float32)
        alpha = (ch_crop[:, :, 3:4].astype(np.float32)) / 255.0
        inv_alpha = 1.0 - alpha
        base_rgb = roi[:, :, :3].astype(np.float32)
        out_rgb = alpha * overlay_rgb + inv_alpha * base_rgb
        roi[:, :, :3] = out_rgb.astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], ch_crop[:, :, 3])
    else:
        roi[:, :, :3] = ch_crop[:, :, :3]
        roi[:, :, 3] = 255


def select_device_and_half(model):
    try:
        import torch
        if torch.cuda.is_available():
            model.to('cuda')
            try:
                model.model.half()
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            return True
        return False
    except Exception:
        return False


def run_inference(model, frame_bgr: np.ndarray, imgsz: int, conf: float, max_det: int, use_half: bool):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(
        rgb,
        verbose=False,
        conf=conf,
        classes=[0],
        imgsz=imgsz,
        max_det=max_det,
        device=0 if use_half else None,
    )
    boxes, confs = [], None
    if results and len(results) > 0:
        res = results[0]
        if hasattr(res, 'boxes') and res.boxes is not None and res.boxes.xyxy is not None:
            boxes = res.boxes.xyxy
            try:
                boxes = boxes.detach().float().cpu().numpy().astype(int)
            except Exception:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            if res.boxes.conf is not None:
                try:
                    confs = res.boxes.conf.detach().float().cpu().numpy()
                except Exception:
                    confs = res.boxes.conf.cpu().numpy()
    return boxes, confs


def main() -> int:
    args = parse_args()

    width, height, fps = args.width, args.height, max(1, args.fps)
    det_color_bgr = parse_bgr_color(args.detection_color)
    cross_color_bgr = parse_bgr_color(args.crosshair_color)

    print(f"🎮 Overlay Publisher Starting...", file=sys.stderr)
    print(f"🖼️  Frame: {width}x{height}@{fps}", file=sys.stderr)
    print(f"🤖 Model: {'disabled' if args.no_detect else args.model}", file=sys.stderr)

    # Load model unless disabled
    model = None
    use_half = False
    if not args.no_detect:
        try:
            from ultralytics import YOLO
            model = YOLO(args.model)
            use_half = select_device_and_half(model)
            print(f"✅ YOLO loaded, FP16={use_half}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}", file=sys.stderr)
            return 1

    # Load crosshair
    crosshair_img = None
    if args.crosshair:
        crosshair_img = load_crosshair_bgra(args.crosshair, args.crosshair_scale, cross_color_bgr)
        if crosshair_img is not None:
            print(f"✅ Crosshair loaded", file=sys.stderr)
        else:
            print(f"⚠️  Crosshair not loaded", file=sys.stderr)

    # Open capture (from shm)
    cap = None
    for _ in range(120):
        cap = cv2.VideoCapture(
            f"shmsrc socket-path={CAPTURE_SOCK} do-timestamp=true is-live=true ! "
            f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
            f"appsink drop=true max-buffers=1 sync=false",
            cv2.CAP_GSTREAMER,
        )
        if cap.isOpened():
            break
        print("⏳ Waiting for capture shm...", file=sys.stderr)
        time.sleep(0.25)
    if not cap or not cap.isOpened():
        print("❌ Failed to open capture shm", file=sys.stderr)
        return 2

    # Open writer (to shm) - BGRA
    writer = cv2.VideoWriter(
        f"appsrc is-live=true do-timestamp=true ! "
        f"video/x-raw,format=BGRA,width={width},height={height},framerate={fps}/1 ! "
        f"queue leaky=downstream max-size-buffers=1 ! "
        f"shmsink socket-path={OVERLAY_SOCK} shm-size=200000000 wait-for-connection=false sync=false async=false",
        cv2.CAP_GSTREAMER,
        0,  # fourcc ignored for gstreamer
        float(fps),
        (width, height),
        True,
    )
    if not writer.isOpened():
        print("❌ Failed to open overlay shm writer", file=sys.stderr)
        return 3

    running = True
    def handle_stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    frame_idx = 0
    last_fps_time = time.time()
    frames_this_sec = 0
    
    # Alpha for rectangles
    det_color_rgba = (det_color_bgr[0], det_color_bgr[1], det_color_bgr[2], 220)

    print("🚀 Overlay loop...", file=sys.stderr)
    while running:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.001)
            continue

        # Prepare transparent overlay RGBA
        overlay = np.zeros((height, width, 4), dtype=np.uint8)

        # Run detection at reduced cadence
        if model is not None and (frame_idx % max(1, args.skip) == 0):
            try:
                boxes, confs = run_inference(model, frame, args.imgsz, args.conf, args.max_det, use_half)
            except Exception as e:
                boxes, confs = [], None
                print(f"⚠️  Detection error: {e}", file=sys.stderr)
        # Draw boxes (persist last boxes by simple reuse)
        if 'boxes' in locals() and boxes is not None and len(boxes) > 0:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), det_color_rgba, 3)
                if (not args.no_label) and confs is not None and i < len(confs):
                    label = f"{confs[i]:.2f}"
                    # Draw label on RGB plane (no alpha blending for text)
                    cv2.putText(overlay, label, (int(x1), max(0, int(y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (det_color_bgr[0], det_color_bgr[1], det_color_bgr[2], 255), 1, cv2.LINE_AA)

        # Crosshair
        if crosshair_img is not None:
            overlay_crosshair_bgra(overlay, crosshair_img)
        else:
            # Simple crosshair as fallback
            cx, cy = width // 2, height // 2
            cv2.line(overlay, (cx - 50, cy), (cx - 12, cy), (*cross_color_bgr, 220), 2)
            cv2.line(overlay, (cx + 12, cy), (cx + 50, cy), (*cross_color_bgr, 220), 2)
            cv2.line(overlay, (cx, cy - 50), (cx, cy - 12), (*cross_color_bgr, 220), 2)
            cv2.line(overlay, (cx, cy + 12), (cx, cy + 50), (*cross_color_bgr, 220), 2)

        # Publish overlay
        writer.write(overlay)

        # FPS log
        frames_this_sec += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            print(f"📊 Publisher FPS: {frames_this_sec:.0f}", file=sys.stderr)
            frames_this_sec = 0
            last_fps_time = now

        frame_idx += 1

    cap.release()
    writer.release()
    print("✅ Overlay publisher stopped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

