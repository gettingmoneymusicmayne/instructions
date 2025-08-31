import argparse
import threading
import time

import numpy as np
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GObject", "2.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GObject, GLib  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("Low-latency GStreamer + YOLO overlay")
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--model", default="yolov11n.pt")
    p.add_argument("--crosshair", default="")
    p.add_argument("--conf", type=float, default=0.4)
    return p.parse_args()


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.boxes = []  # list of (x1,y1,x2,y2,conf)
        self.crosshair_bgra = None  # numpy BGRA
        self.frame_shape = (1080, 1920)  # default


def start_detection_thread(appsink, state: SharedState, model_path: str, conf: float):
    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)

    def worker():
        while True:
            sample = appsink.emit("try-pull-sample", 100000)
            if sample is None:
                continue
            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            _, mapinfo = buf.map(Gst.MapFlags.READ)
            try:
                array = np.frombuffer(mapinfo.data, dtype=np.uint8)
                frame = array.reshape((height, width, 3))  # BGR
            except Exception:
                buf.unmap(mapinfo)
                continue
            # Update frame shape for overlay centering
            with state.lock:
                state.frame_shape = (height, width)
            # Run detection on RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, verbose=False, conf=conf, classes=[0])
            new_boxes = []
            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.xyxy is not None:
                b = results[0].boxes
                xyxy = b.xyxy.cpu().numpy()
                confs = b.conf.cpu().numpy() if b.conf is not None else None
                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = xyxy[i].astype(int)
                    c = float(confs[i]) if confs is not None else 0.0
                    new_boxes.append((x1, y1, x2, y2, c))
            with state.lock:
                state.boxes = new_boxes
            buf.unmap(mapinfo)

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def load_crosshair_bgra(path: str):
    import cv2
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        # Ensure BGRA
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img
    except Exception:
        return None


def main():
    args = parse_args()
    Gst.init(None)
    GObject.threads_init()

    # Prepare multiple pipeline options to handle different device formats, using compositor+appsrc
    pipeline_options = []
    # 1) Raw caps locked to your device's advertised uncompressed mode
    pipeline_options.append(
        (
            "raw",
            (
                f"v4l2src device={args.device} io-mode=0 ! "
                f"video/x-raw,format=YUY2,width={args.width},height={args.height},framerate={args.fps}/1 ! "
                f"queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGRA ! tee name=t "
                # Display branch via compositor (sink_0)
                f"t. ! queue leaky=downstream max-size-buffers=2 ! compositor name=comp sink_0::zorder=0 ! xvimagesink sync=false "
                # Detection branch to appsink (BGR)
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink drop=true max-buffers=1 emit-signals=true sync=false "
                # Appsrc overlay branch into compositor (sink_1)
                f"appsrc name=boxes is-live=true format=time caps=video/x-raw,format=BGRA,width={args.width},height={args.height},framerate={args.fps}/1 ! queue ! comp."
            ),
        )
    )
    # 2) MJPEG decode path
    pipeline_options.append(
        (
            "mjpeg",
            (
                f"v4l2src device={args.device} io-mode=0 ! "
                f"image/jpeg,width={args.width},height={args.height},framerate={args.fps}/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGRA ! tee name=t "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! compositor name=comp sink_0::zorder=0 ! xvimagesink sync=false "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink drop=true max-buffers=1 emit-signals=true sync=false "
                f"appsrc name=boxes is-live=true format=time caps=video/x-raw,format=BGRA,width={args.width},height={args.height},framerate={args.fps}/1 ! queue ! comp."
            ),
        )
    )
    # 3) Minimal constraints
    pipeline_options.append(
        (
            "auto",
            (
                f"v4l2src device={args.device} io-mode=0 ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGRA ! tee name=t "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! compositor name=comp sink_0::zorder=0 ! xvimagesink sync=false "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink drop=true max-buffers=1 emit-signals=true sync=false "
                f"appsrc name=boxes is-live=true format=time caps=video/x-raw,format=BGRA,width={args.width},height={args.height},framerate={args.fps}/1 ! queue ! comp."
            ),
        )
    )

    pipeline = None
    chosen = None
    overlay = None
    appsink = None
    for name, desc in pipeline_options:
        try:
            candidate = Gst.parse_launch(desc)
            # Probe by setting to PAUSED first
            ret = candidate.set_state(Gst.State.PAUSED)
            if ret == Gst.StateChangeReturn.FAILURE:
                candidate.set_state(Gst.State.NULL)
                continue
            # Wait briefly for negotiation
            bus = candidate.get_bus()
            msg = bus.timed_pop_filtered(2_000_000_000, Gst.MessageType.ERROR | Gst.MessageType.ASYNC_DONE | Gst.MessageType.STATE_CHANGED)
            if msg and msg.type == Gst.MessageType.ERROR:
                candidate.set_state(Gst.State.NULL)
                continue
            # Success; finalize
            pipeline = candidate
            overlay = pipeline.get_by_name("overlay")
            appsink = pipeline.get_by_name("appsink")
            chosen = name
            break
        except Exception:
            try:
                candidate.set_state(Gst.State.NULL)
            except Exception:
                pass
            continue
    if pipeline is None:
        print("Failed to build a working pipeline (raw/mjpeg/auto). Check device and plugins.")
        return 5

    state = SharedState()

    # Load crosshair image
    if args.crosshair:
        state.crosshair_bgra = load_crosshair_bgra(args.crosshair)

    # Detection thread
    start_detection_thread(appsink, state, args.model, args.conf)

    # Get elements
    comp = pipeline.get_by_name("comp")
    appsrc_boxes = pipeline.get_by_name("boxes")

    # Configure appsrc timing
    appsrc_boxes.set_property("is-live", True)
    appsrc_boxes.set_property("do-timestamp", True)
    # Push overlay frames periodically
    import cv2

    def make_overlay_frame() -> bytes:
        with state.lock:
            fh, fw = state.frame_shape
            boxes = list(state.boxes)
            cross = state.crosshair_bgra
        if fw <= 0 or fh <= 0:
            fw, fh = 1920, 1080
        # Transparent BGRA frame
        frame = np.zeros((fh, fw, 4), dtype=np.uint8)
        # Draw boxes in yellow with 80% alpha
        for (x1, y1, x2, y2, _c) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255, 200), 3)
        # Overlay crosshair centered if present
        if cross is not None:
            ch = cross
            ch_h, ch_w = ch.shape[:2]
            x = max(0, (fw - ch_w) // 2)
            y = max(0, (fh - ch_h) // 2)
            x2 = min(fw, x + ch_w)
            y2 = min(fh, y + ch_h)
            cw_eff = x2 - x
            ch_eff = y2 - y
            if cw_eff > 0 and ch_eff > 0:
                roi = frame[y:y2, x:x2]
                ch_crop = ch[0:ch_eff, 0:cw_eff]
                if ch_crop.shape[2] == 4:
                    overlay_rgb = ch_crop[:, :, :3].astype(np.float32)
                    alpha = (ch_crop[:, :, 3:4].astype(np.float32)) / 255.0
                    inv_alpha = 1.0 - alpha
                    base_rgb = roi[:, :, :3].astype(np.float32)
                    out_rgb = alpha * overlay_rgb + inv_alpha * base_rgb
                    roi[:, :, :3] = out_rgb.astype(np.uint8)
                    # Set alpha to max of existing and overlay alpha
                    roi[:, :, 3] = np.maximum(roi[:, :, 3], ch_crop[:, :, 3])
                else:
                    roi[:, :, :3] = ch_crop[:, :, :3]
                    roi[:, :, 3] = 255
        return frame.tobytes()

    def push_loop():
        fps = max(1, args.fps)
        frame_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, fps)
        timestamp = 0
        while True:
            data = make_overlay_frame()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.pts = timestamp
            buf.dts = timestamp
            buf.duration = frame_duration
            ret = appsrc_boxes.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                time.sleep(0.01)
            timestamp += frame_duration
            time.sleep(0.001)

    threading.Thread(target=push_loop, daemon=True).start()

    # Run
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message):  # noqa: ARG001
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print("ERROR:", err, dbg)
            pipeline.set_state(Gst.State.NULL)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            pipeline.set_state(Gst.State.NULL)
            loop.quit()

    bus.connect("message", on_message)

    loop = GLib.MainLoop()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    raise SystemExit(main())

