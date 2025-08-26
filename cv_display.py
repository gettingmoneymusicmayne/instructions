#!/usr/bin/env python3
"""
Gaming Overlay System - AI Detection with Custom Crosshair
Handles video capture, YOLO person detection, and crosshair overlay with customizable colors.
Optimized for Jetson: CUDA/FP16 when available, optional TensorRT engine, input resizing, and frame skipping.
"""

import argparse
import signal
import time
import sys
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Gaming Overlay - AI Detection with Crosshair")
    parser.add_argument("--device", default="/dev/video0", help="Video capture device")
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1080, help="Capture height")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--model", default="yolov11n.pt", help="YOLO model path or .engine")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (short side)")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections per image")
    parser.add_argument("--skip", type=int, default=2, help="Run detection every N frames (>=1)")
    parser.add_argument("--crosshair", default="", help="Crosshair image path")
    parser.add_argument("--crosshair-color", default="0,255,0", help="Crosshair color (BGR)")
    parser.add_argument("--detection-color", default="0,255,255", help="Detection box color (BGR)")
    parser.add_argument("--crosshair-scale", type=float, default=1.0, help="Crosshair scale")
    parser.add_argument("--no-label", action="store_true", help="Do not draw confidence labels")
    return parser.parse_args()


def parse_bgr_color(color_str: str) -> tuple:
    """Parse BGR color string to tuple."""
    try:
        b, g, r = map(int, color_str.split(','))
        return (b, g, r)
    except Exception:
        return (0, 255, 0)  # Default green


def load_crosshair_bgra(path: str, scale: float = 1.0, color: tuple = (0, 255, 0)):
    """Load crosshair image and optionally recolor it."""
    if not path or not os.path.exists(path):
        return None
    
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        
        # Ensure BGRA format
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # Recolor the crosshair if needed
        if color != (0, 255, 0):  # If not default green
            alpha = img[:, :, 3]
            mask = alpha > 0
            img[mask, 0] = color[0]  # Blue
            img[mask, 1] = color[1]  # Green
            img[mask, 2] = color[2]  # Red
        
        # Scale if needed
        if scale != 1.0:
            new_w = max(1, int(img.shape[1] * scale))
            new_h = max(1, int(img.shape[0] * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img
    except Exception as e:
        print(f"Error loading crosshair: {e}", file=sys.stderr)
        return None


def overlay_crosshair_image(frame: np.ndarray, crosshair_bgra: np.ndarray) -> None:
    """Overlay crosshair image onto frame with proper alpha blending."""
    if crosshair_bgra is None:
        return
    
    fh, fw = frame.shape[:2]
    ch, cw = crosshair_bgra.shape[:2]
    
    # Center the crosshair
    x = max(0, (fw - cw) // 2)
    y = max(0, (fh - ch) // 2)
    x2 = min(fw, x + cw)
    y2 = min(fh, y + ch)
    
    cw_eff = x2 - x
    ch_eff = y2 - y
    
    if cw_eff <= 0 or ch_eff <= 0:
        return
    
    roi = frame[y:y2, x:x2]
    ch_crop = crosshair_bgra[0:ch_eff, 0:cw_eff]
    
    if ch_crop.shape[2] == 4:
        # BGRA image with alpha channel
        overlay_rgb = ch_crop[:, :, :3].astype(np.float32)
        alpha = (ch_crop[:, :, 3:4].astype(np.float32)) / 255.0
        inv_alpha = 1.0 - alpha
        base_rgb = roi[:, :, :3].astype(np.float32)
        out_rgb = alpha * overlay_rgb + inv_alpha * base_rgb
        roi[:, :, :3] = out_rgb.astype(np.uint8)
    else:
        # BGR image without alpha
        roi[:, :, :3] = ch_crop[:, :, :3]


def draw_crosshair_simple(frame: np.ndarray, color: tuple, thickness: int = 2, gap: int = 10, length: int = 50) -> None:
    """Draw a simple crosshair using lines."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Horizontal left
    cv2.line(frame, (cx - length, cy), (cx - gap, cy), color, thickness)
    # Horizontal right
    cv2.line(frame, (cx + gap, cy), (cx + length, cy), color, thickness)
    # Vertical top
    cv2.line(frame, (cx, cy - length), (cx, cy - gap), color, thickness)
    # Vertical bottom
    cv2.line(frame, (cx, cy + gap), (cx, cy + length), color, thickness)


def select_device_and_dtype(model):
    """Configure device (CUDA if available) and dtype (FP16 when possible)."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            model.to('cuda')
            # Half precision speeds up on Jetson GPUs
            try:
                model.model.half()
            except Exception:
                pass
            # Enable CUDNN autotuner
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            return 'cuda', True
        else:
            return 'cpu', False
    except Exception:
        return 'cpu', False


def run_inference(model, frame_bgr: np.ndarray, imgsz: int, conf: float, max_det: int, use_half: bool):
    """Run YOLO inference on a frame; returns list of boxes and confs."""
    import cv2  # local for safety
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Ultralytics handles resizing; we pass numpy directly
    results = model.predict(
        rgb,
        verbose=False,
        conf=conf,
        classes=[0],  # person
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
    
    # Parse colors
    crosshair_color = parse_bgr_color(args.crosshair_color)
    detection_color = parse_bgr_color(args.detection_color)
    
    print(f"🎮 Gaming Overlay Starting...", file=sys.stderr)
    print(f"📹 Device: {args.device}", file=sys.stderr)
    print(f"🎯 Crosshair: {args.crosshair}", file=sys.stderr)
    print(f"🤖 Model: {args.model}", file=sys.stderr)
    print(f"🖼️  Inference size: {args.imgsz}, Skip: {args.skip}", file=sys.stderr)
    
    # Load YOLO model (TensorRT engine .engine is supported by Ultralytics)
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"✅ YOLO model loaded: {args.model}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}", file=sys.stderr)
        return 1
    
    # Select device and dtype
    device, use_half = select_device_and_dtype(model)
    print(f"🧠 Device: {device}, FP16: {use_half}", file=sys.stderr)
    
    # Load crosshair
    crosshair_img = None
    if args.crosshair:
        crosshair_img = load_crosshair_bgra(args.crosshair, args.crosshair_scale, crosshair_color)
        if crosshair_img is not None:
            print(f"✅ Crosshair loaded: {args.crosshair}", file=sys.stderr)
        else:
            print(f"⚠️  Crosshair not loaded, will use simple crosshair", file=sys.stderr)
    
    # Setup video capture with multiple fallback methods
    cap = None
    
    # Method 1: GStreamer pipeline (best for Jetson)
    try:
        pipeline = (
            f"v4l2src device={args.device} io-mode=0 ! "
            f"video/x-raw,format=YUY2,width={args.width},height={args.height},framerate={args.fps}/1 ! "
            f"queue leaky=downstream max-size-buffers=1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"✅ Using GStreamer pipeline", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  GStreamer failed: {e}", file=sys.stderr)
    
    # Method 2: Direct V4L2
    if not cap or not cap.isOpened():
        try:
            cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                cap.set(cv2.CAP_PROP_FPS, args.fps)
                print(f"✅ Using V4L2 direct capture", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  V4L2 failed: {e}", file=sys.stderr)
    
    # Method 3: Fallback to device 0
    if not cap or not cap.isOpened():
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                cap.set(cv2.CAP_PROP_FPS, args.fps)
                print(f"✅ Using fallback capture (device 0)", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Fallback failed: {e}", file=sys.stderr)
    
    if not cap or not cap.isOpened():
        print(f"❌ Failed to open video capture", file=sys.stderr)
        return 1
    
    # Setup display window
    cv2.namedWindow("Gaming Overlay", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaming Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Signal handling for clean shutdown
    running = True
    def handle_stop(signum, frame):
        nonlocal running
        running = False
        print(f"\n🛑 Shutting down...", file=sys.stderr)
    
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    
    # Performance tracking
    frame_count = 0
    last_fps_time = time.time()
    
    print(f"🚀 Starting overlay loop...", file=sys.stderr)
    
    # Cache last detections to reuse on skipped frames
    last_boxes, last_confs = [], None
    
    # Ensure valid skip
    skip_n = max(1, int(args.skip))
    
    while running:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.001)
            continue
        
        # Decide whether to run detection this frame
        do_detect = (frame_count % skip_n) == 0
        if do_detect:
            try:
                boxes, confs = run_inference(model, frame, args.imgsz, args.conf, args.max_det, use_half)
                last_boxes, last_confs = boxes, confs
            except Exception as e:
                # Keep previous detections if inference fails transiently
                print(f"⚠️  Detection error: {e}", file=sys.stderr)
        
        # Draw detection boxes (use last known if skipping)
        if last_boxes is not None and len(last_boxes) > 0:
            for i, (x1, y1, x2, y2) in enumerate(last_boxes):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), detection_color, 2)
                if (not args.no_label) and last_confs is not None and i < len(last_confs):
                    conf_text = f"{last_confs[i]:.2f}"
                    cv2.putText(frame, conf_text, (int(x1), max(0, int(y1) - 6)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color, 1, cv2.LINE_AA)
        
        # Draw crosshair
        if crosshair_img is not None:
            overlay_crosshair_image(frame, crosshair_img)
        else:
            draw_crosshair_simple(frame, crosshair_color, thickness=2, gap=12, length=50)
        
        # Display frame
        cv2.imshow("Gaming Overlay", frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q to quit
            break
        
        # FPS calculation
        frame_count += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frame_count / (now - last_fps_time)
            print(f"📊 FPS: {fps:.1f}", file=sys.stderr)
            frame_count = 0
            last_fps_time = now
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Overlay stopped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

