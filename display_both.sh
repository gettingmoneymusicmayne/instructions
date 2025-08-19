#!/bin/bash

# Usage: ./display_both.sh /dev/videoX 1920 1080 60
# Publishes capture frames to /tmp/capture_bgr and reads overlay frames from /tmp/overlay_bgra
# Shows the composed result (capture + overlay) with very low latency.

set -euo pipefail

DEVICE="${1:-/dev/video0}"
WIDTH="${2:-1920}"
HEIGHT="${3:-1080}"
FPS="${4:-60}"

# Clean old sockets
rm -f /tmp/capture_bgr /tmp/overlay_bgra || true
mkfifo /tmp/overlay_bgra || true

gst-launch-1.0 -e \
  v4l2src device="$DEVICE" io-mode=0 ! \
  video/x-raw,format=YUY2,width=$WIDTH,height=$HEIGHT,framerate=$FPS/1 ! \
  queue leaky=downstream max-size-buffers=1 ! \
  videoconvert ! video/x-raw,format=BGRA ! \
  tee name=t \
  t. ! queue leaky=downstream max-size-buffers=1 ! compositor name=comp sink_0::zorder=0 ! videoconvert ! xvimagesink sync=false \
  t. ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=BGR ! shmsink socket-path=/tmp/capture_bgr shm-size=200000000 wait-for-connection=false sync=false async=false \
  shmsrc socket-path=/tmp/overlay_bgra is-live=true do-timestamp=true ! video/x-raw,format=BGRA,width=$WIDTH,height=$HEIGHT,framerate=$FPS/1 ! queue ! comp.

