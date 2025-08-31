#!/bin/bash

# Usage: ./display_tee.sh /dev/videoX /path/to/crosshair.png OFFSET_X OFFSET_Y 1920 1080 60
# Displays low-latency video with crosshair and publishes BGR frames to /tmp/capture_bgr for detection.

set -euo pipefail

DEVICE="${1:-/dev/video0}"
CROSSHAIR="${2:-}"
OX="${3:-928}"
OY="${4:-508}"
WIDTH="${5:-1920}"
HEIGHT="${6:-1080}"
FPS="${7:-60}"

# Validate inputs
if [ ! -e "$DEVICE" ]; then
  echo "Video device not found: $DEVICE" >&2
  exit 1
fi

if [ -n "$CROSSHAIR" ] && [ ! -f "$CROSSHAIR" ]; then
  echo "Crosshair image not found: $CROSSHAIR" >&2
  exit 1
fi

rm -f /tmp/capture_bgr || true

# Build pipeline with optional crosshair
if [ -n "$CROSSHAIR" ]; then
  gst-launch-1.0 -e \
    v4l2src device="$DEVICE" io-mode=0 ! \
    'video/x-raw,format=YUY2,width='"$WIDTH"',height='"$HEIGHT"',framerate='"$FPS"'/1' ! \
    queue leaky=downstream max-size-buffers=1 ! \
    videoconvert ! tee name=t \
    t. ! queue leaky=downstream max-size-buffers=1 ! gdkpixbufoverlay location="$CROSSHAIR" offset-x=$OX offset-y=$OY ! xvimagesink sync=false \
    t. ! queue leaky=downstream max-size-buffers=1 ! video/x-raw,format=BGR ! shmsink socket-path=/tmp/capture_bgr shm-size=200000000 wait-for-connection=false sync=false async=false
else
  gst-launch-1.0 -e \
    v4l2src device="$DEVICE" io-mode=0 ! \
    'video/x-raw,format=YUY2,width='"$WIDTH"',height='"$HEIGHT"',framerate='"$FPS"'/1' ! \
    queue leaky=downstream max-size-buffers=1 ! \
    videoconvert ! tee name=t \
    t. ! queue leaky=downstream max-size-buffers=1 ! xvimagesink sync=false \
    t. ! queue leaky=downstream max-size-buffers=1 ! video/x-raw,format=BGR ! shmsink socket-path=/tmp/capture_bgr shm-size=200000000 wait-for-connection=false sync=false async=false
fi

