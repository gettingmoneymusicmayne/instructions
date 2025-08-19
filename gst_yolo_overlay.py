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
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--crosshair", default="")
    p.add_argument("--conf", type=float, default=0.4)
    return p.parse_args()


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.boxes = []  # list of (x1,y1,x2,y2,conf)
        self.crosshair_surface = None  # cairo surface
        self.frame_shape = (0, 0)


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


def load_crosshair_surface(path: str):
    import cairo
    try:
        return cairo.ImageSurface.create_from_png(path)
    except Exception:
        return None


def main():
    args = parse_args()
    Gst.init(None)
    GObject.threads_init()

    # Prepare multiple pipeline options to handle different device formats
    pipeline_options = []
    # 1) Raw caps (let kernel choose raw pixel format)
    pipeline_options.append(
        (
            "raw",
            (
                f"v4l2src device={args.device} io-mode=2 ! "
                f"video/x-raw,format=YUY2,width={args.width},height={args.height},framerate={args.fps}/1 ! "
                f"queue leaky=downstream max-size-buffers=2 ! tee name=t "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGRA,width={args.width},height={args.height} ! cairooverlay name=overlay ! xvimagesink sync=false "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR,width={args.width},height={args.height},framerate={args.fps}/1 ! appsink name=appsink caps=video/x-raw,format=BGR,width={args.width},height={args.height},framerate={args.fps}/1 drop=true max-buffers=1 emit-signals=true sync=false"
            ),
        )
    )
    # 2) MJPEG decode path
    pipeline_options.append(
        (
            "mjpeg",
            (
                f"v4l2src device={args.device} io-mode=2 ! "
                f"image/jpeg,width={args.width},height={args.height},framerate={args.fps}/1 ! jpegdec ! "
                f"queue leaky=downstream max-size-buffers=2 ! tee name=t "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGRA,width={args.width},height={args.height} ! cairooverlay name=overlay ! xvimagesink sync=false "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR,width={args.width},height={args.height},framerate={args.fps}/1 ! appsink name=appsink caps=video/x-raw,format=BGR,width={args.width},height={args.height},framerate={args.fps}/1 drop=true max-buffers=1 emit-signals=true sync=false"
            ),
        )
    )
    # 3) Minimal constraints
    pipeline_options.append(
        (
            "auto",
            (
                f"v4l2src device={args.device} io-mode=2 ! queue leaky=downstream max-size-buffers=2 ! tee name=t "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGRA ! cairooverlay name=overlay ! xvimagesink sync=false "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink drop=true max-buffers=1 emit-signals=true sync=false"
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

    # Load crosshair surface
    if args.crosshair:
        state.crosshair_surface = load_crosshair_surface(args.crosshair)

    # Detection thread
    start_detection_thread(appsink, state, args.model, args.conf)

    # cairooverlay callback
    def on_draw(overlay_obj, context, timestamp, duration):  # noqa: ARG001
        import cairo
        # Draw boxes
        with state.lock:
            boxes = list(state.boxes)
            fh, fw = state.frame_shape
            cross = state.crosshair_surface
        context.set_source_rgba(1.0, 1.0, 0.0, 0.9)
        context.set_line_width(3.0)
        for x1, y1, x2, y2, conf in boxes:
            context.rectangle(x1, y1, x2 - x1, y2 - y1)
            context.stroke()
        # Draw crosshair image centered
        if cross is not None and fw > 0 and fh > 0:
            ch_w = cross.get_width()
            ch_h = cross.get_height()
            x = max(0, (fw - ch_w) // 2)
            y = max(0, (fh - ch_h) // 2)
            context.set_source_surface(cross, x, y)
            context.paint()

    overlay.connect("draw", on_draw)

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

