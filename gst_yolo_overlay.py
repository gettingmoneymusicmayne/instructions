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
        self.crosshair_bgra = None  # numpy BGRA
        self.frame_shape = (1080, 1920)  # default
        self.overlay_pipeline = None # Added for overlay pipeline
        self.enhanced_pipeline = None # Added for enhanced pipeline
        self.detection_sink = None # Added for enhanced pipeline


def start_detection_thread(appsink, state: SharedState, model_path: str, conf: float):
    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)

    def worker():
        while True:
            # Use enhanced detection sink if available, otherwise fall back to original
            current_sink = getattr(state, 'detection_sink', None) or appsink
            
            sample = current_sink.emit("try-pull-sample", 100000)
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

    # Simplified pipeline that prevents "unknown" window issues
    # Use a single display path with proper window naming
    pipeline_options = []
    
    # Option 1: Direct display with appsink for detection
    pipeline_options.append(
        (
            "direct",
            (
                f"v4l2src device={args.device} io-mode=0 ! "
                f"video/x-raw,format=YUY2,width={args.width},height={args.height},framerate={args.fps}/1 ! "
                f"queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGRA ! "
                f"tee name=t "
                # Display branch - direct to xvimagesink with proper naming
                f"t. ! queue leaky=downstream max-size-buffers=2 ! "
                f"xvimagesink name=display sync=false "
                # Detection branch to appsink (BGR)
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR ! "
                f"appsink name=appsink drop=true max-buffers=1 emit-signals=true sync=false"
            ),
        )
    )
    
    # Option 2: Fallback with MJPEG support
    pipeline_options.append(
        (
            "mjpeg",
            (
                f"v4l2src device={args.device} io-mode=0 ! "
                f"image/jpeg,width={args.width},height={args.height},framerate={args.fps}/1 ! jpegdec ! "
                f"videoconvert ! video/x-raw,format=BGRA ! "
                f"tee name=t "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! "
                f"xvimagesink name=display sync=false "
                f"t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR ! "
                f"appsink name=appsink drop=true max-buffers=1 emit-signals=true sync=false"
            ),
        )
    )

    pipeline = None
    chosen = None
    appsink = None
    display_sink = None
    
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
            display_sink = pipeline.get_by_name("display")
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
        print("Failed to build a working pipeline. Check device and plugins.")
        return 5

    state = SharedState()

    # Load crosshair image
    if args.crosshair:
        state.crosshair_bgra = load_crosshair_bgra(args.crosshair)

    # Detection thread
    start_detection_thread(appsink, state, args.model, args.conf)

    # Configure display sink properties to prevent unwanted windows
    if display_sink:
        try:
            # Set window properties to prevent "unknown" window
            display_sink.set_property("name", "YOLO_Display")
            # Force fullscreen and above other windows
            display_sink.set_property("force-aspect-ratio", False)
        except Exception:
            pass

    # Create a modified pipeline that draws overlays directly on the video
    # This prevents the "unknown" window by using a single display path
    print("Creating enhanced display pipeline with overlays...")
    
    # Stop the current pipeline
    pipeline.set_state(Gst.State.NULL)
    
    # Create new pipeline with overlay drawing
    enhanced_pipeline_desc = (
        f"v4l2src device={args.device} io-mode=0 ! "
        f"video/x-raw,format=YUY2,width={args.width},height={args.height},framerate={args.fps}/1 ! "
        f"queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR ! "
        f"tee name=t "
        # Display branch with OpenCV processing
        f"t. ! queue leaky=downstream max-size-buffers=2 ! "
        f"videoconvert ! video/x-raw,format=RGB ! "
        f"appsink name=display_sink drop=true max-buffers=1 emit-signals=true sync=false "
        # Detection branch
        f"t. ! queue leaky=downstream max-size-buffers=2 ! "
        f"appsink name=detection_sink drop=true max-buffers=1 emit-signals=true sync=false"
    )
    
    try:
        enhanced_pipeline = Gst.parse_launch(enhanced_pipeline_desc)
        display_sink = enhanced_pipeline.get_by_name("display_sink")
        detection_sink = enhanced_pipeline.get_by_name("detection_sink")
        
        # Start the enhanced pipeline
        enhanced_pipeline.set_state(Gst.State.PLAYING)
        
        # Wait for pipeline to start
        time.sleep(0.5)
        
        # Function to process frames with overlays
        def process_display_frames():
            import cv2
            
            while True:
                try:
                    # Get frame from display sink
                    sample = display_sink.emit("try-pull-sample", 100000)  # 100ms
                    if sample is None:
                        time.sleep(0.001)
                        continue
                    
                    buf = sample.get_buffer()
                    caps = sample.get_caps()
                    structure = caps.get_structure(0)
                    width = structure.get_value('width')
                    height = structure.get_value('height')
                    
                    _, mapinfo = buf.map(Gst.MapFlags.READ)
                    try:
                        array = np.frombuffer(mapinfo.data, dtype=np.uint8)
                        frame = array.reshape((height, width, 3))  # RGB
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except Exception:
                        buf.unmap(mapinfo)
                        continue
                    
                    # Draw YOLO detection boxes
                    with state.lock:
                        boxes = list(state.boxes)
                        cross = state.crosshair_bgra
                    
                    # Draw boxes in yellow
                    for (x1, y1, x2, y2, _c) in boxes:
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    
                    # Overlay crosshair centered if present
                    if cross is not None:
                        ch_h, ch_w = cross.shape[:2]
                        x = max(0, (width - ch_w) // 2)
                        y = max(0, (height - ch_h) // 2)
                        x2 = min(width, x + ch_w)
                        y2 = min(height, y + ch_h)
                        cw_eff = x2 - x
                        ch_eff = y2 - y
                        if cw_eff > 0 and ch_eff > 0:
                            roi = frame_bgr[y:y2, x:x2]
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
                    
                    # Display the frame with OpenCV (this will be the main display)
                    cv2.imshow("YOLO Detection", frame_bgr)
                    cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
                    
                    # Handle key events
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  # ESC or 'q'
                        break
                    
                    buf.unmap(mapinfo)
                    
                except Exception as e:
                    print(f"Display processing error: {e}")
                    time.sleep(0.01)
        
        # Start display processing thread
        display_thread = threading.Thread(target=process_display_frames, daemon=True)
        display_thread.start()
        
        # Store enhanced pipeline for cleanup
        state.enhanced_pipeline = enhanced_pipeline
        
        # Update detection thread to use new sink
        state.detection_sink = detection_sink
        
    except Exception as e:
        print(f"Failed to create enhanced pipeline: {e}")
        print("Falling back to basic display...")
        # Restart original pipeline
        pipeline.set_state(Gst.State.PLAYING)
        state.enhanced_pipeline = None

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
    
    # Wait a moment for the pipeline to start
    time.sleep(0.5)
    
    # Try to set window properties after pipeline is running
    if display_sink:
        try:
            # Use wmctrl to find and control the window
            import subprocess
            import time
            
            # Wait for window to appear
            for _ in range(10):
                try:
                    result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, timeout=2)
                    if 'YOLO_Display' in result.stdout or 'gst-launch' in result.stdout:
                        break
                except Exception:
                    pass
                time.sleep(0.2)
            
            # Try to set fullscreen and above other windows
            try:
                subprocess.run(['wmctrl', '-r', 'YOLO_Display', '-b', 'add,fullscreen,above'], 
                             capture_output=True, timeout=2)
            except Exception:
                # Fallback: try to find any gst-launch window
                try:
                    result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, timeout=2)
                    for line in result.stdout.splitlines():
                        if 'gst-launch' in line:
                            window_id = line.split()[0]
                            subprocess.run(['wmctrl', '-i', '-r', window_id, '-b', 'add,fullscreen,above'], 
                                         capture_output=True, timeout=2)
                            break
                except Exception:
                    pass
        except Exception:
            pass
    
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        if state.enhanced_pipeline:
            state.enhanced_pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    raise SystemExit(main())

