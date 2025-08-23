# YOLO Overlay "Unknown Window" Fix

## Problem Description

The original `gst_yolo_overlay.py` script was creating an "unknown" window on NVIDIA Jetson Orin Nano when running YOLO detection with crosshair overlay. This happened because:

1. **Complex GStreamer Pipeline**: The original script used a `compositor` element with multiple sink inputs
2. **Multiple Display Elements**: The pipeline created separate display paths that could spawn unexpected windows
3. **Appsrc Overlay**: The `appsrc` for drawing boxes sometimes created its own display window

## Root Cause

The "unknown" window appeared due to:
- GStreamer's `compositor` element creating multiple display contexts
- Complex pipeline with multiple sinks and sources
- Window naming conflicts between different pipeline elements

## Solution

The fix involves simplifying the GStreamer pipeline to use a single display path:

### Before (Problematic):
```bash
# Complex pipeline with compositor
v4l2src ! ... ! compositor name=comp sink_0::zorder=0 ! xvimagesink
# Separate overlay path
appsrc name=boxes ! ... ! comp.
```

### After (Fixed):
```bash
# Simple pipeline with single display path
v4l2src ! ... ! tee name=t
t. ! ... ! videoconvert ! appsink name=display_sink  # For OpenCV display
t. ! ... ! videoconvert ! appsink name=detection_sink # For YOLO detection
```

## Key Changes

1. **Removed Compositor**: Eliminated the `compositor` element that was causing window conflicts
2. **Single Display Path**: All overlays are now drawn using OpenCV and displayed in one window
3. **Proper Window Naming**: The main display window is explicitly named "YOLO Detection"
4. **Simplified Pipeline**: Reduced complexity to prevent unexpected window creation

## Benefits

- ✅ **No More "Unknown" Window**: Single, properly named display window
- ✅ **Better Performance**: Simplified pipeline with fewer elements
- ✅ **Easier Debugging**: Clearer pipeline structure
- ✅ **Fullscreen Support**: Proper fullscreen behavior maintained
- ✅ **Crosshair + YOLO**: Both features work together seamlessly

## Usage

```bash
# Run with YOLO detection and crosshair
python3 gst_yolo_overlay.py --crosshair crosshair.png --model yolo11n.pt

# Run with just crosshair (no YOLO)
python3 gst_yolo_overlay.py --crosshair crosshair.png

# Run with just YOLO (no crosshair)
python3 gst_yolo_overlay.py --model yolo11n.pt
```

## Testing

Use the provided test script to verify the fix:

```bash
python3 test_yolo_fix.py
```

This will test both the crosshair-only mode (known working) and the YOLO+crosshair mode (previously problematic).

## Technical Details

The fix works by:
1. Capturing video through GStreamer
2. Processing frames with OpenCV for overlays
3. Displaying the final result in a single OpenCV window
4. Maintaining the GStreamer pipeline for video capture only

This approach eliminates the GStreamer display complexity that was causing the "unknown" window issue while preserving all functionality.