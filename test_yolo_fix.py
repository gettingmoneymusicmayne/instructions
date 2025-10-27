#!/usr/bin/env python3
"""
Test script for the fixed YOLO overlay system.
This script tests the basic functionality without requiring a capture device.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        print("✓ GStreamer bindings imported successfully")
    except Exception as e:
        print(f"✗ GStreamer bindings failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except Exception as e:
        print(f"✗ OpenCV failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except Exception as e:
        print(f"✗ NumPy failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
    except Exception as e:
        print(f"✗ Ultralytics YOLO failed: {e}")
        print("  Install with: pip3 install ultralytics")
        return False
    
    return True

def test_gstreamer():
    """Test if GStreamer is working properly."""
    print("\nTesting GStreamer...")
    
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        
        Gst.init(None)
        print("✓ GStreamer initialized successfully")
        
        # Test a simple pipeline
        pipeline_desc = "videotestsrc ! fakesink"
        pipeline = Gst.parse_launch(pipeline_desc)
        if pipeline:
            print("✓ GStreamer pipeline parsing works")
            pipeline.set_state(Gst.State.NULL)
        else:
            print("✗ GStreamer pipeline parsing failed")
            return False
            
    except Exception as e:
        print(f"✗ GStreamer test failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """Test if YOLO model can be loaded."""
    print("\nTesting YOLO model...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load a small model (this will download if not present)
        print("Loading YOLO model (this may take a moment)...")
        model = YOLO("yolo11n.pt")
        print("✓ YOLO model loaded successfully")
        
        # Test basic inference
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(dummy_image, verbose=False, conf=0.4, classes=[0])
        print("✓ YOLO inference test passed")
        
    except Exception as e:
        print(f"✗ YOLO model test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("YOLO Overlay System Test")
    print("=" * 40)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing dependencies.")
        return 1
    
    # Test 2: GStreamer
    if not test_gstreamer():
        print("\n❌ GStreamer tests failed. Please check GStreamer installation.")
        return 1
    
    # Test 3: YOLO Model
    if not test_yolo_model():
        print("\n❌ YOLO model tests failed. Please check YOLO installation.")
        return 1
    
    print("\n✅ All tests passed!")
    print("\nThe system should work without creating 'unknown' windows.")
    print("\nTo run the actual overlay system:")
    print("python3 gst_yolo_overlay.py --device /dev/video0 --crosshair crosshair.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())