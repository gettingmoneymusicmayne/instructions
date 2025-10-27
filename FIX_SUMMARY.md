# YOLO Overlay "Unknown" Window Fix

## Problem Description

The original `gst_yolo_overlay.py` script was creating an unwanted "unknown" window when running both YOLO detection and crosshair overlay. This happened because:

1. **Complex GStreamer Pipeline**: The original script used a `compositor` element with multiple sink inputs
2. **Multiple Display Elements**: The pipeline created separate display paths that could spawn multiple windows
3. **Appsrc Overlay**: The `appsrc` element for drawing boxes sometimes created its own display window
4. **Window Naming Conflicts**: GStreamer elements weren't properly named, leading to generic "unknown" window titles

## Root Cause

The issue was in this problematic pipeline structure:
```bash
compositor name=comp sink_0::zorder=0 ! xvimagesink sync=false
# ... plus appsrc overlay branch into compositor (sink_1)
appsrc name=boxes ! ... ! comp.
```

This approach:
- Created multiple display sinks
- Used complex compositor logic
- Could spawn unexpected windows
- Had poor window naming control

## Solution Implemented

### 1. Simplified Pipeline Structure
Replaced the complex compositor-based pipeline with a simpler approach:
```bash
v4l2src ! ... ! tee name=t
t. ! ... ! videoconvert ! video/x-raw,format=RGB ! appsink name=display_sink
t. ! ... ! videoconvert ! video/x-raw,format=BGR ! appsink name=detection_sink
```

### 2. Single Display Window
- Uses OpenCV (`cv2.imshow`) for the main display
- Single window named "YOLO Detection" 
- Proper fullscreen handling
- No multiple GStreamer display sinks

### 3. Unified Overlay Drawing
- YOLO boxes and crosshair are drawn directly on the video frame
- Single processing thread handles all overlays
- No separate overlay pipeline that could create extra windows

### 4. Proper Window Management
- Explicit window naming
- Fullscreen property setting
- Fallback to basic display if enhanced pipeline fails

## Key Changes Made

1. **Removed `compositor` element** - Eliminated the source of multiple display paths
2. **Simplified pipeline structure** - Single video path with tee for detection/display
3. **Unified overlay system** - All drawing happens in one place with OpenCV
4. **Better error handling** - Graceful fallback to basic display mode
5. **Proper cleanup** - Both pipelines are properly stopped on exit

## Benefits

- ✅ **No more "unknown" windows**
- ✅ **Single, properly named display window**
- ✅ **Maintains full YOLO detection functionality**
- ✅ **Keeps crosshair overlay working**
- ✅ **Better performance with simplified pipeline**
- ✅ **More reliable window management**

## Usage

The fixed script works the same way as before:

```bash
# Basic usage
python3 gst_yolo_overlay.py --device /dev/video0 --crosshair crosshair.png

# With custom model
python3 gst_yolo_overlay.py --device /dev/video0 --crosshair crosshair.png --model yolo11n.pt

# With custom confidence threshold
python3 gst_yolo_overlay.py --device /dev/video0 --crosshair crosshair.png --conf 0.5
```

## Testing

Run the test script to verify everything works:
```bash
python3 test_yolo_fix.py
```

This will test:
- All required imports
- GStreamer functionality  
- YOLO model loading
- Basic inference

## Technical Details

The fix maintains the same external API while internally:
- Using a single GStreamer pipeline for video capture
- Processing frames through OpenCV for overlays
- Displaying the final result in one named window
- Properly managing the detection thread with the new pipeline structure

This approach eliminates the window management issues while preserving all the functionality of the original system.