#!/usr/bin/env python3
"""
Gaming Overlay - Low-latency render + decoupled AI (latest-frame, lock-free)

- Render loop does NOT sleep; it runs as fast as the capture/display allow (smoother).
- AI thread runs independently at --ai-fps, always consumes the latest frame (deque maxlen=1).
- AI uses pre-resized snapshot to --imgsz to reduce per-frame overhead.
- Low-latency GStreamer pipeline (leaky queue, drop=true, sync=false).
- Short persistence to keep boxes snappy without flicker.
"""

import argparse
import signal
import time
import sys
import os
import threading
from collections import deque

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Gaming Overlay - Low-latency YOLO + Crosshair")
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--model", default="yolov11n.pt")  # set to yolov11n.engine for max FPS
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--ai-fps", type=float, default=24.0)
    p.add_argument("--max-det", type=int, default=30)
    p.add_argument("--persist-ms", type=int, default=150)
    p.add_argument("--crosshair", default="")
    p.add_argument("--crosshair-color", default="0,255,0")
    p.add_argument("--detection-color", default="0,255,255")
    p.add_argument("--crosshair-scale", type=float, default=1.0)
    p.add_argument("--no-label", action="store_true")
    p.add_argument("--no-detect", action="store_true")
    return p.parse_args()


def parse_bgr(s: str) -> tuple:
    try:
        b, g, r = map(int, s.split(','))
        return (b, g, r)
    except Exception:
        return (0, 255, 0)


def load_crosshair_bgra(path: str, scale: float, color: tuple):
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
    except Exception:
        return None


def overlay_crosshair(frame: np.ndarray, crosshair_bgra: np.ndarray, color_fallback: tuple):
    if crosshair_bgra is not None:
        fh, fw = frame.shape[:2]
        ch, cw = crosshair_bgra.shape[:2]
        x = max(0, (fw - cw) // 2); y = max(0, (fh - ch) // 2)
        x2 = min(fw, x + cw); y2 = min(fh, y + ch)
        cw_eff = x2 - x; ch_eff = y2 - y
        if cw_eff <= 0 or ch_eff <= 0:
            return
        roi = frame[y:y2, x:x2]
        ch_crop = crosshair_bgra[0:ch_eff, 0:cw_eff]
        if ch_crop.shape[2] == 4:
            overlay_rgb = ch_crop[:, :, :3].astype(np.float32)
            alpha = (ch_crop[:, :, 3:4].astype(np.float32)) / 255.0
            inv_alpha = 1.0 - alpha
            base_rgb = roi[:, :, :3].astype(np.float32)
            out_rgb = alpha * overlay_rgb + inv_alpha * base_rgb
            roi[:, :, :3] = out_rgb.astype(np.uint8)
        else:
            roi[:, :, :3] = ch_crop[:, :, :3]
    else:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 50, cy), (cx - 12, cy), color_fallback, 2)
        cv2.line(frame, (cx + 12, cy), (cx + 50, cy), color_fallback, 2)
        cv2.line(frame, (cx, cy - 50), (cx, cy - 12), color_fallback, 2)
        cv2.line(frame, (cx, cy + 12), (cx, cy + 50), color_fallback, 2)


def select_device_half(model):
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


def model_predict(model, bgr_resized: np.ndarray, conf: float, max_det: int, use_half: bool):
    # bgr_resized is already imgsz x imgsz
    rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    results = model.predict(
        rgb,
        verbose=False,
        conf=conf,
        classes=[0],
        imgsz=bgr_resized.shape[0],
        max_det=max_det,
        device=0 if use_half else None,
    )
    boxes, confs = [], None
    if results and len(results) > 0:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None and res.boxes.xyxy is not None:
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


def inference_worker(stop_event: threading.Event,
                     latest_frame: deque,
                     latest_det: dict,
                     model, use_half: bool,
                     conf: float, max_det: int,
                     imgsz: int, ai_fps: float):
    period = max(0.0, 1.0 / max(1.0, ai_fps))
    target_size = (imgsz, imgsz)
    while not stop_event.is_set():
        t0 = time.time()

        if not latest_frame:
            time.sleep(0.001)
            continue
        # Consume most recent frame only (lock-free latest)
        frame = latest_frame[-1]
        # Resize snapshot to imgsz to reduce conversion cost
        snap = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        try:
            boxes, confs = model_predict(model, snap, conf, max_det, use_half)
        except Exception as e:
            boxes, confs = [], None
            print(f"⚠️  Detection error: {e}", file=sys.stderr)

        latest_det["boxes"] = boxes
        latest_det["confs"] = confs
        latest_det["t_ms"] = int(time.time() * 1000)

        # Pace AI
        elapsed = time.time() - t0
        sleep_s = max(0.0, period - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)


def main() -> int:
    args = parse_args()

    # OpenCV optimizations
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)
    except Exception:
        pass

    # Low-latency capture pipeline
    pipeline = (
        f"v4l2src device={args.device} io-mode=0 ! "
        f"video/x-raw,format=YUY2,width={args.width},height={args.height},framerate={args.fps}/1 ! "
        f"queue leaky=downstream max-size-buffers=1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ GStreamer capture failed. Falling back to V4L2", file=sys.stderr)
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("❌ Failed to open video capture", file=sys.stderr)
            return 4
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Window
    cv2.namedWindow("Gaming Overlay", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaming Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    det_color = parse_bgr(args.detection_color)
    cross_color = parse_bgr(args.crosshair_color)
    crosshair_img = load_crosshair_bgra(args.crosshair, args.crosshair_scale, cross_color)

    # Load model
    model = None
    use_half = False
    if not args.no_detect:
        try:
            from ultralytics import YOLO
            model = YOLO(args.model)
            # Warm-up once on imgsz-sized tensor
            _ = model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8), imgsz=args.imgsz, verbose=False)
            use_half = select_device_half(model)
            print(f"✅ Model loaded: {args.model}, FP16={use_half}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to load model: {e}", file=sys.stderr)
            return 2

    # Shared latest buffers (lock-free, single-producer single-consumer pattern)
    latest_frame = deque(maxlen=1)   # producer: render loop; consumer: AI thread
    latest_det = {"boxes": [], "confs": None, "t_ms": 0}

    stop_event = threading.Event()
    worker = None
    if model is not None:
        worker = threading.Thread(
            target=inference_worker,
            args=(stop_event, latest_frame, latest_det, model, use_half,
                  args.conf, args.max_det, args.imgsz, args.ai_fps),
            daemon=True
        )
        worker.start()

    # Signals
    running = True
    def handle_stop(signum, frame):
        nonlocal running
        running = False
        stop_event.set()
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    last_log = time.time()
    frames = 0
    print("🚀 Low-latency render loop + decoupled AI...", file=sys.stderr)

    while running:
        ok, frame = cap.read()
        if not ok or frame is None:
            # No sleep to keep responsiveness; minimal backoff
            continue

        # Publish latest frame for AI (no lock, overwrite latest)
        if model is not None:
            latest_frame.append(frame)

            # Draw last detections if still fresh
            t_ms = latest_det.get("t_ms", 0)
            if t_ms > 0 and (int(time.time() * 1000) - t_ms) <= args.persist_ms:
                boxes = latest_det.get("boxes", [])
                confs = latest_det.get("confs", None)
                if boxes is not None and len(boxes) > 0:
                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), det_color, 2)
                        if not args.no_label and confs is not None and i < len(confs):
                            cv2.putText(frame, f"{confs[i]:.2f}", (int(x1), max(0, int(y1) - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 1, cv2.LINE_AA)

        # Crosshair
        overlay_crosshair(frame, crosshair_img, cross_color)

        cv2.imshow("Gaming Overlay", frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break

        frames += 1
        now = time.time()
        if now - last_log >= 1.0:
            print(f"📊 Render FPS: {frames:.0f}", file=sys.stderr)
            frames = 0
            last_log = now

    stop_event.set()
    if worker is not None:
        worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Stopped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

