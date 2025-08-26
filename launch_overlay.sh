#!/bin/bash

# Gaming Overlay Launcher - Crosshair Only Mode
# Usage: ./launch_overlay.sh /path/to/crosshair.png [offset_x] [offset_y] [device]
# Requires: gstreamer1.0-tools, gstreamer1.0-plugins-good, wmctrl, xdotool, unclutter

set -euo pipefail

CROSSHAIR_PATH="${1:-}" 
OFFSET_X="${2:-948}"
OFFSET_Y="${3:-528}"
DEVICE="${4:-/dev/video0}"

echo "🎮 Gaming Overlay - Crosshair Mode Starting..." >&2

# Validate inputs
if [ -z "$CROSSHAIR_PATH" ] || [ ! -f "$CROSSHAIR_PATH" ]; then
  echo "❌ Crosshair image not found or not provided: $CROSSHAIR_PATH" >&2
  echo "💡 Please provide a valid crosshair.png file" >&2
  exit 1
fi

if [ ! -e "$DEVICE" ]; then
  echo "❌ Video device not found: $DEVICE" >&2
  echo "💡 Check your capture card connection" >&2
  exit 1
fi

echo "✅ Crosshair: $CROSSHAIR_PATH" >&2
echo "✅ Device: $DEVICE" >&2
echo "✅ Position: ($OFFSET_X, $OFFSET_Y)" >&2

# Kill any previous gst-launch instances
echo "🔄 Stopping previous overlays..." >&2
pkill -f "gst-launch-1.0" 2>/dev/null || true
sleep 1

# Hide the mouse cursor after a second of inactivity
echo "🖱️  Hiding mouse cursor..." >&2
unclutter >/dev/null 2>&1 &

# Start the GStreamer pipeline with crosshair overlay
echo "🚀 Starting GStreamer overlay..." >&2
gst-launch-1.0 \
  v4l2src device="$DEVICE" ! \
  "video/x-raw,format=YUY2,width=1920,height=1080,framerate=60/1" ! \
  videoconvert ! \
  gdkpixbufoverlay location="$CROSSHAIR_PATH" offset-x=$OFFSET_X offset-y=$OFFSET_Y ! \
  xvimagesink sync=false \
  >/dev/null 2>&1 &

# Give it time to start
sleep 2

# Make the window fullscreen and above others
echo "🖥️  Configuring display..." >&2
for i in {1..15}; do
  WINDOW_ID=$(wmctrl -l | grep "gst-launch-1.0" | awk '{print $1}')
  if [ -n "$WINDOW_ID" ]; then
    echo "✅ Found overlay window, making fullscreen..." >&2
    wmctrl -i -r "$WINDOW_ID" -b add,fullscreen
    wmctrl -i -r "$WINDOW_ID" -b add,above
    wmctrl -i -r "$WINDOW_ID" -b remove,demands_attention
    break
  fi
  sleep 0.2
done

# Move cursor out of the way
echo "🖱️  Moving cursor to corner..." >&2
xdotool mousemove 0 0 >/dev/null 2>&1 || true

echo "✅ Crosshair overlay active!" >&2
echo "💡 Press Ctrl+C to stop" >&2

# Wait for interrupt
trap 'echo "🛑 Stopping overlay..."; pkill -f "gst-launch-1.0" 2>/dev/null || true; exit 0' INT TERM
wait

