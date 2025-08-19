import os
import sys
import time
import signal
import argparse

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO detection overlay with optional crosshair")
    parser.add_argument("--device", type=str, default="/dev/video0", help="Video capture device path")
    parser.add_argument("--model", type=str, default="yolov11n.pt", help="Ultralytics model path")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--crosshair-enable", type=int, default=1, help="Enable crosshair (1/0)")
    parser.add_argument("--crosshair-color", type=str, default="#00ff00", help="Crosshair color hex, e.g. #00ff00")
    parser.add_argument("--window-title", type=str, default="Jetson Overlay", help="OpenCV window title")
    parser.add_argument("--fullscreen", type=int, default=1, help="Fullscreen window (1/0)")
    parser.add_argument("--width", type=int, default=1920, help="Target capture width")
    parser.add_argument("--height", type=int, default=1080, help="Target capture height")
    parser.add_argument("--fps", type=int, default=60, help="Target capture FPS")
    return parser.parse_args()


def hex_to_bgr(color_hex: str) -> tuple:
    hh = color_hex.lstrip('#')
    r, g, b = tuple(int(hh[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def build_gst_pipeline(device: str, width: int, height: int, fps: int) -> str:
    # Convert to BGR for OpenCV
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true sync=false"
    )


def draw_crosshair(frame: np.ndarray, color_bgr: tuple, thickness: int = 2, gap: int = 10, length: int = 50) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    # Horizontal left
    cv2.line(frame, (cx - length, cy), (cx - gap, cy), color_bgr, thickness)
    # Horizontal right
    cv2.line(frame, (cx + gap, cy), (cx + length, cy), color_bgr, thickness)
    # Vertical top
    cv2.line(frame, (cx, cy - length), (cx, cy - gap), color_bgr, thickness)
    # Vertical bottom
    cv2.line(frame, (cx, cy + gap), (cx, cy + length), color_bgr, thickness)


def main() -> int:
    args = parse_args()

    # Lazy import ultralytics to allow environments without it to start
    try:
        from ultralytics import YOLO
    except Exception as ex:  # noqa: BLE001
        print("Failed to import ultralytics. Install with: pip3 install ultralytics", file=sys.stderr)
        print(str(ex), file=sys.stderr)
        return 2

    # Load model
    try:
        model = YOLO(args.model)
    except Exception as ex:  # noqa: BLE001
        print(f"Failed to load model {args.model}: {ex}", file=sys.stderr)
        return 3

    # Prefer GStreamer pipeline for reliable formats
    pipeline = build_gst_pipeline(args.device, args.width, args.height, args.fps)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        # Fallback to V4L2 direct
        print("GStreamer capture failed, falling back to V4L2", file=sys.stderr)
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("Failed to open video capture", file=sys.stderr)
            return 4
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Window setup
    cv2.namedWindow(args.window_title, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(args.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    color_bgr = hex_to_bgr(args.crosshair_color)
    running = True

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    last_infer = 0.0
    target_interval = 1.0 / max(1, args.fps)

    while running:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue

        # Inference (convert BGR->RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, verbose=False, conf=args.conf, classes=[0])  # class 0 = person

        # Draw boxes
        if results and len(results) > 0:
            res = results[0]
            if hasattr(res, "boxes") and res.boxes is not None and res.boxes.xyxy is not None:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    if confs is not None:
                        label = f"person {confs[i]:.2f}"
                        cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw crosshair if enabled
        if args.crosshair_enable:
            draw_crosshair(frame, color_bgr=color_bgr, thickness=2, gap=12, length=50)

        cv2.imshow(args.window_title, frame)

        # Keep UI responsive
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        # Simple pacing
        now = time.time()
        if now - last_infer < target_interval:
            time.sleep(max(0.0, target_interval - (now - last_infer)))
        last_infer = now

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

