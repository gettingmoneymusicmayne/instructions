#!/bin/bash

# Ultra low-latency 120Hz display for Jetson with UVC capture cards
# Usage: ./launch_120hz.sh [device] [width] [height] [fps]
# Example: ./launch_120hz.sh /dev/video0 1920 1080 120

set -euo pipefail

DEVICE="${1:-/dev/video0}"
WIDTH="${2:-1920}"
HEIGHT="${3:-1080}"
FPS="${4:-120}"

if [ ! -e "$DEVICE" ]; then
  echo "❌ Video device not found: $DEVICE" >&2
  exit 1
fi

# Prefer Jetson EGL sink
SINK="xvimagesink"
if gst-inspect-1.0 nveglglessink >/dev/null 2>&1; then
  SINK="nveglglessink"
fi

echo "🎯 Low-latency 120Hz path: $DEVICE ${WIDTH}x${HEIGHT}@${FPS} -> $SINK" >&2

# Notes:
# - io-mode=2 enables DMABUF for v4l2src where supported
# - Keep NV12 throughout, convert to NVMM via nvvidconv for zero-copy to sink
# - sync=false and qos=false minimize latency (may allow tearing)

gst-launch-1.0 -e \
  v4l2src device="$DEVICE" io-mode=2 ! \
  "video/x-raw,format=NV12,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1" ! \
  queue leaky=downstream max-size-buffers=1 ! \
  nvvidconv ! "video/x-raw(memory:NVMM),format=NV12" ! \
  ${SINK} sync=false qos=false

