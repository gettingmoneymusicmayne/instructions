import argparse
import signal
import time
import threading

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Low-latency CV display with YOLO boxes and crosshair")
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--crosshair", default="")
    p.add_argument("--imgsz", type=int, default=512, help="Inference size (short side), e.g., 320/416/512/640")
    # Accept BOTH hyphen and underscore forms; normalize to compute_device
    p.add_argument("--compute-device", dest="compute_device", default="0",
                   help="Ultralytics device: 0 for CUDA GPU, 'cpu' for CPU")
    p.add_argument("--compute_device", dest="compute_device", default="0", help=argparse.SUPPRESS)
    p.add_argument("--half", type=int, default=1, help="Use FP16 on CUDA (1/0)")
    p.add_argument("--display_backend", choices=["opencv", "gstreamer"], default="gstreamer",
                   help="Display backend for final output (gstreamer reduces tearing vs OpenCV)")
    p.add_argument("--gst_sink", default="glimagesink",
                   help="GStreamer sink (glimagesink, xvimagesink, nveglglessink). Default glimagesink")
    p.add_argument("--gst_sync", type=int, default=0,
                   help="GStreamer sink sync flag (0=no vsync for lowest latency, 1=vsync for smoothness)")
    return p.parse_args()


def load_crosshair_bgra(path: str):
    if not path:
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as ex:
        print("Install ultralytics: pip3 install ultralytics")
        print(ex)
        return 2

    model = YOLO(args.model)

    pipeline = (
        f"v4l2src device={args.device} io-mode=0 ! "
        f"video/x-raw,format=YUY2,width={args.width},height={args.height},framerate={args.fps}/1 ! "
        f"queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open capture with GStreamer pipeline. Falling back to V4L2.")
        # Try device path first, then index 0
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                print("Failed to open video capture")
                return 4
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Prepare display backend
    writer = None
    use_opencv_display = (getattr(args, "display_backend", "gstreamer") == "opencv")
    if use_opencv_display:
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        sync_flag = "true" if int(getattr(args, "gst_sync", 0)) else "false"
        gst_display_pipeline = (
            f"appsrc is-live=true do-timestamp=true block=true ! "
            f"video/x-raw,format=BGR,width={args.width},height={args.height},framerate={args.fps}/1 ! "
            f"queue leaky=downstream max-size-buffers=1 ! videoconvert ! {args.gst_sink} sync={sync_flag}"
        )
        writer = cv2.VideoWriter(gst_display_pipeline, cv2.CAP_GSTREAMER, 0, float(args.fps), (args.width, args.height))
        if not writer.isOpened():
            # Fallback to OpenCV window
            writer = None
            use_opencv_display = True
            cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Try to minimize internal buffering (may be ignored depending on backend)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    cross = load_crosshair_bgra(args.crosshair)

    running = True

    # Shared state between capture/display thread (main) and inference thread
    shared_lock = threading.Lock()
    latest_det_input = {"img": None, "scale_x": 1.0, "scale_y": 1.0}
    latest_boxes = []  # list of (x1,y1,x2,y2)

    def handle_stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    def inference_worker():
        nonlocal running, latest_det_input, latest_boxes
        predict_kwargs = {
            "verbose": False,
            "conf": args.conf,
            "classes": [0],
            "imgsz": args.imgsz,
        }
        # Device/precision preferences
        if args.compute_device.lower() != "cpu":
            predict_kwargs["device"] = 0
            if args.half:
                predict_kwargs["half"] = True
        else:
            predict_kwargs["device"] = "cpu"
        while running:
            # Get the most recent small image to process
            with shared_lock:
                det_img = latest_det_input["img"]
                sx = latest_det_input["scale_x"]
                sy = latest_det_input["scale_y"]
                latest_det_input["img"] = None
            if det_img is None:
                time.sleep(0.001)
                continue
            # Run inference
            det_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
            results = model.predict(det_rgb, **predict_kwargs)
            boxes_scaled = []
            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.xyxy is not None:
                b = results[0].boxes
                xyxy = b.xyxy.cpu().numpy()
                for (x1, y1, x2, y2) in xyxy:
                    # Scale back to full-res coordinates
                    boxes_scaled.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
            with shared_lock:
                latest_boxes = boxes_scaled

    t = threading.Thread(target=inference_worker, daemon=True)
    t.start()

    # Simple frame pacing to make output cadence consistent when not using vsync
    target_dt = 1.0 / float(args.fps)
    next_deadline = time.monotonic()

    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.001)
            continue

        # Prepare a downscaled frame for detection (decoupled from display)
        h, w = frame.shape[:2]
        target_w = max(160, min(args.imgsz, 1280))
        if w <= target_w:
            small = frame
            sx = 1.0
            sy = 1.0
        else:
            scale = target_w / float(w)
            det_w = target_w
            det_h = max(1, int(h * scale))
            small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
            sx = 1.0 / scale
            sy = 1.0 / scale

        # Publish latest detection input (size-1 mailbox)
        with shared_lock:
            latest_det_input["img"] = small
            latest_det_input["scale_x"] = sx
            latest_det_input["scale_y"] = sy

        # Draw last known boxes without blocking for inference
        with shared_lock:
            boxes_snapshot = list(latest_boxes)
        for (x1, y1, x2, y2) in boxes_snapshot:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Overlay crosshair centered
        if cross is not None:
            h, w = frame.shape[:2]
            ch_h, ch_w = cross.shape[:2]
            x = max(0, (w - ch_w) // 2)
            y = max(0, (h - ch_h) // 2)
            x2 = min(w, x + ch_w)
            y2 = min(h, y + ch_h)
            cw_eff = x2 - x
            ch_eff = y2 - y
            if cw_eff > 0 and ch_eff > 0:
                roi = frame[y:y2, x:x2]
                ch_crop = cross[0:ch_eff, 0:cw_eff]
                if ch_crop.shape[2] == 4:
                    overlay_rgb = ch_crop[:, :, :3].astype(np.float32)
                    alpha = (ch_crop[:, :, 3:4].astype(np.float32)) / 255.0
                    inv_alpha = 1.0 - alpha
                    base_rgb = roi[:, :, :3].astype(np.float32)
                    out_rgb = alpha * overlay_rgb + inv_alpha * base_rgb
                    roi[:, :, :3] = out_rgb.astype(np.uint8)
                else:
                    roi[:, :, :3] = ch_crop[:, :, :3]

        if use_opencv_display:
            cv2.imshow("Display", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        else:
            # Frame pacing without vsync
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(max(0.0, next_deadline - now))
            writer.write(frame)
            next_deadline += target_dt

    cap.release()
    if use_opencv_display:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

