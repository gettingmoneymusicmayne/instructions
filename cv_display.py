import argparse
import signal
import time

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Low-latency CV display with YOLO boxes and crosshair")
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--model", default="yolov11n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--crosshair", default="")
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
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("Failed to open video capture")
            return 4
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cross = load_crosshair_bgra(args.crosshair)

    running = True

    def handle_stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.001)
            continue

        # YOLO inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, verbose=False, conf=args.conf, classes=[0])
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.xyxy is not None:
            b = results[0].boxes
            xyxy = b.xyxy.cpu().numpy().astype(int)
            for (x1, y1, x2, y2) in xyxy:
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

        cv2.imshow("Display", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

