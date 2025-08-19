import argparse
import time
import sys
import signal

import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser("YOLO detection from shm, overlay to shm")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--crosshair", default="")
    p.add_argument("--crosshair-scale", type=float, default=1.0)
    return p.parse_args()


def load_crosshair_bgra(path: str, scale: float):
    if not path:
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if scale != 1.0:
        new_w = max(1, int(img.shape[1] * scale))
        new_h = max(1, int(img.shape[0] * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as ex:
        print("Install ultralytics: pip3 install ultralytics", file=sys.stderr)
        print(ex, file=sys.stderr)
        return 2

    model = YOLO(args.model)

    width, height = args.width, args.height
    fps = max(1, args.fps)

    # SHM read (capture frames)
    cap = None
    while True:
        cap = cv2.VideoCapture(
            "shmsrc socket-path=/tmp/capture_bgr do-timestamp=true is-live=true ! "
            "video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! appsink drop=true max-buffers=1" % (width, height, fps),
            cv2.CAP_GSTREAMER,
        )
        if cap.isOpened():
            break
        print("Waiting for capture shared memory /tmp/capture_bgr ...", file=sys.stderr)
        time.sleep(0.3)

    # SHM write (overlay frames)
    # We will paint the overlay into the display by drawing directly onto the captured frame copy
    # and publishing it back is not needed since display draws crosshair already. Instead, we keep
    # overlay separate and display reads only crosshair. For now, we will open a debug preview off.
    writer = None

    cross = load_crosshair_bgra(args.crosshair, args.crosshair_scale)

    running = True
    def handle_stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.002)
            continue

        overlay = np.zeros((height, width, 4), dtype=np.uint8)

        # YOLO detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, verbose=False, conf=args.conf, classes=[0])
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.xyxy is not None:
            b = results[0].boxes
            xyxy = b.xyxy.cpu().numpy().astype(int)
            for (x1, y1, x2, y2) in xyxy:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255, 220), 3)

        # Crosshair overlay centered
        if cross is not None:
            ch_h, ch_w = cross.shape[:2]
            x = max(0, (width - ch_w) // 2)
            y = max(0, (height - ch_h) // 2)
            x2 = min(width, x + ch_w)
            y2 = min(height, y + ch_h)
            cw_eff = x2 - x
            ch_eff = y2 - y
            if cw_eff > 0 and ch_eff > 0:
                roi = overlay[y:y2, x:x2]
                ch_crop = cross[0:ch_eff, 0:cw_eff]
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

        # No writer path; detection-only side effects for now

    cap.release()
    if writer is not None:
        writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

