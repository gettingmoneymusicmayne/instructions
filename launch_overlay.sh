#!/bin/bash

# Usage: ./launch_overlay.sh /path/to/crosshair.png [offset_x] [offset_y]
# Requires: gstreamer1.0-tools, gstreamer1.0-plugins-good, wmctrl, xdotool, unclutter

set -euo pipefail

CROSSHAIR_PATH="${1:-}" 
OFFSET_X="${2:-948}"
OFFSET_Y="${3:-528}"
if [ -z "$CROSSHAIR_PATH" ] || [ ! -f "$CROSSHAIR_PATH" ]; then
  echo "Crosshair image not found or not provided: $CROSSHAIR_PATH" >&2
  exit 1
fi

# Kill any previous gst-launch instances quietly
pkill -f "gst-launch-1.0" 2>/dev/null || true

# Hide the mouse cursor after a second of inactivity
unclutter >/dev/null 2>&1 &

# Start the GStreamer pipeline with the specified crosshair overlay
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  "video/x-raw,format=YUY2,width=1920,height=1080,framerate=60/1" ! \
  videoconvert ! \
  gdkpixbufoverlay location="$CROSSHAIR_PATH" offset-x=$OFFSET_X offset-y=$OFFSET_Y ! \
  xvimagesink sync=false \
  >/dev/null 2>&1 &

# Give it time to start
sleep 1

# Make the window fullscreen and above others (best-effort)
for i in {1..10}; do
  WINDOW_ID=$(wmctrl -l | grep "gst-launch-1.0" | awk '{print $1}')
  if [ -n "$WINDOW_ID" ]; then
    wmctrl -i -r "$WINDOW_ID" -b add,fullscreen
    wmctrl -i -r "$WINDOW_ID" -b add,above
    wmctrl -i -r "$WINDOW_ID" -b remove,demands_attention
    break
  fi
  sleep 0.2
done

# Move cursor out of the way
xdotool mousemove 0 0 >/dev/null 2>&1 || true

