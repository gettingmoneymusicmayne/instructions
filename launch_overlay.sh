#!/bin/bash

# Gaming Overlay Launcher - Composite Mode (Base video + RGBA overlay via SHM)
# Usage: ./launch_overlay.sh [device] [width] [height] [fps]
# Requires: gstreamer1.0-tools, gstreamer1.0-plugins-good, wmctrl, xdotool, unclutter

set -euo pipefail

DEVICE="${1:-/dev/video0}"
WIDTH="${2:-1920}"
HEIGHT="${3:-1080}"
FPS="${4:-60}"

CAPTURE_SOCK="/tmp/capture_bgr"
OVERLAY_SOCK="/tmp/overlay_rgba"

echo "🎮 Gaming Overlay - Composite Mode Starting..." >&2

if [ ! -e "$DEVICE" ]; then
  echo "❌ Video device not found: $DEVICE" >&2
  echo "💡 Check your capture card connection" >&2
  exit 1
fi

echo "✅ Device: $DEVICE  ${WIDTH}x${HEIGHT}@${FPS}" >&2

# Choose sink: prefer nveglglessink on Jetson, fallback to xvimagesink
SINK="xvimagesink"
if gst-inspect-1.0 nveglglessink >/dev/null 2>&1; then
  SINK="nveglglessink"
fi

echo "🧩 Using sink: $SINK" >&2

# Stop previous overlays and remove old sockets
pkill -f "gst-launch-1.0" 2>/dev/null || true
rm -f "$CAPTURE_SOCK" "$OVERLAY_SOCK" || true
sleep 0.5

# Hide the mouse cursor after a second of inactivity
unclutter >/dev/null 2>&1 &

# Launch pipeline:
# - Tee 1: publish BGR frames to /tmp/capture_bgr for AI process
# - Tee 2: composite base video with /tmp/overlay_rgba (BGRA) and display at 60fps

echo "🚀 Starting GStreamer tee + compositor pipeline..." >&2

gst-launch-1.0 -e \
  v4l2src device="$DEVICE" io-mode=0 ! \
  "video/x-raw,format=YUY2,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1" ! \
  queue leaky=downstream max-size-buffers=1 ! \
  videoconvert ! video/x-raw,format=BGR ! \
  tee name=t \
  t. ! queue leaky=downstream max-size-buffers=1 ! \
      video/x-raw,format=BGR,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 ! \
      shmsink socket-path=${CAPTURE_SOCK} shm-size=300000000 wait-for-connection=true sync=false async=false \
  t. ! queue leaky=downstream max-size-buffers=1 ! \
      videoconvert ! video/x-raw,format=BGRA ! \
      compositor name=comp background=1 \
      ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! ${SINK} sync=false \
  shmsrc socket-path=${OVERLAY_SOCK} do-timestamp=true is-live=true \
      ! video/x-raw,format=BGRA,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 \
      ! queue leaky=downstream max-size-buffers=1 ! comp. \
  &

# Give it time to start
sleep 1.5

# Fullscreen the window if we are using an X sink
if [ "$SINK" = "xvimagesink" ]; then
  for i in {1..15}; do
    WINDOW_ID=$(wmctrl -l | grep "gst-launch-1.0" | awk '{print $1}')
    if [ -n "$WINDOW_ID" ]; then
      wmctrl -i -r "$WINDOW_ID" -b add,fullscreen
      wmctrl -i -r "$WINDOW_ID" -b add,above
      wmctrl -i -r "$WINDOW_ID" -b remove,demands_attention
      break
    fi
    sleep 0.2
  done
fi

echo "✅ Display pipeline active at 60fps (render), waiting for overlay publisher..." >&2

echo "💡 Overlay publisher should write BGRA frames to: ${OVERLAY_SOCK}" >&2

echo "Press Ctrl+C to stop." >&2
trap 'echo "🛑 Stopping overlay..."; pkill -f "gst-launch-1.0" 2>/dev/null || true; exit 0' INT TERM
wait

