#!/usr/bin/env python3
"""
Test script to verify the setup and dependencies are working correctly.
"""

import os
import sys
import subprocess
import importlib.util

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    modules = [
        ('flask', 'Flask'),
        ('PIL', 'PIL'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
    ]
    
    for module_name, display_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"✓ {display_name} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {display_name}: {e}")
            return False
    
    # Test ultralytics separately as it's optional for basic functionality
    try:
        importlib.import_module('ultralytics')
        print("✓ Ultralytics imported successfully")
    except ImportError:
        print("⚠ Ultralytics not available (YOLO detection will not work)")
    
    return True

def test_files():
    """Test if required files exist."""
    print("\nTesting files...")
    
    files = [
        'app.py',
        'detector.py',
        'launch_overlay.sh',
        'display_tee.sh',
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            return False
    
    return True

def test_video_devices():
    """Test if video devices are available."""
    print("\nTesting video devices...")
    
    devices = ['/dev/video0', '/dev/video1']
    found_devices = []
    
    for device in devices:
        if os.path.exists(device):
            print(f"✓ {device} exists")
            found_devices.append(device)
        else:
            print(f"⚠ {device} not found")
    
    if not found_devices:
        print("⚠ No video devices found - capture may not work")
        return False
    
    return True

def test_gstreamer():
    """Test if GStreamer is available."""
    print("\nTesting GStreamer...")
    
    try:
        result = subprocess.run(['gst-launch-1.0', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ GStreamer is available")
            return True
        else:
            print("✗ GStreamer command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ GStreamer not found or not working")
        return False

def test_web_ui():
    """Test if the web UI can start."""
    print("\nTesting web UI...")
    
    try:
        # Test if Flask app can be imported
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        print("✓ Flask app can be imported")
        return True
    except Exception as e:
        print(f"✗ Failed to import Flask app: {e}")
        return False

def main():
    """Run all tests."""
    print("Crosshair Dashboard Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_files,
        test_video_devices,
        test_gstreamer,
        test_web_ui,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Setup looks good.")
        return 0
    else:
        print("⚠ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())