#!/usr/bin/env python3
"""
Test script to verify YOLO detection and crosshair overlay without "unknown" window
"""

import subprocess
import time
import signal
import sys

def test_yolo_overlay():
    """Test the fixed YOLO overlay script"""
    
    print("Testing YOLO overlay with crosshair...")
    print("This should display video with YOLO detection boxes and crosshair")
    print("Press ESC or 'q' to exit")
    print()
    
    # Check if we have the required files
    import os
    if not os.path.exists("gst_yolo_overlay.py"):
        print("Error: gst_yolo_overlay.py not found!")
        return False
    
    if not os.path.exists("crosshair.png"):
        print("Warning: crosshair.png not found. Running without crosshair...")
        crosshair_arg = ""
    else:
        crosshair_arg = "crosshair.png"
    
    # Build command
    cmd = [
        "python3", "gst_yolo_overlay.py",
        "--device", "/dev/video0",
        "--width", "1920",
        "--height", "1080", 
        "--fps", "60",
        "--model", "yolo11n.pt",
        "--crosshair", crosshair_arg,
        "--conf", "0.4"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        # Start the process
        process = subprocess.Popen(cmd)
        
        print("Process started. Waiting for display...")
        time.sleep(3)  # Give it time to start
        
        # Check if process is still running
        if process.poll() is None:
            print("✓ Process is running successfully")
            print("✓ No 'unknown' window should appear")
            print("✓ Video should display with YOLO detection and crosshair")
            print()
            print("Press Enter to stop the test...")
            input()
        else:
            print("✗ Process failed to start")
            return False
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Clean up
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        print("Process stopped")
    
    return True

def test_crosshair_only():
    """Test just the crosshair overlay (known working)"""
    
    print("Testing crosshair-only overlay...")
    print("This should work without issues")
    print()
    
    if not os.path.exists("launch_overlay.sh"):
        print("Error: launch_overlay.sh not found!")
        return False
    
    if not os.path.exists("crosshair.png"):
        print("Error: crosshair.png not found!")
        return False
    
    try:
        cmd = ["./launch_overlay.sh", "crosshair.png", "948", "528", "/dev/video0"]
        print(f"Running: {' '.join(cmd)}")
        print()
        
        process = subprocess.Popen(cmd)
        time.sleep(3)
        
        if process.poll() is None:
            print("✓ Crosshair overlay is running")
            print("Press Enter to stop...")
            input()
        else:
            print("✗ Crosshair overlay failed")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        print("Crosshair overlay stopped")
    
    return True

if __name__ == "__main__":
    print("YOLO Overlay Test Suite")
    print("=" * 40)
    print()
    
    # Test crosshair only first (known working)
    print("1. Testing crosshair-only overlay...")
    if test_crosshair_only():
        print("✓ Crosshair test passed\n")
    else:
        print("✗ Crosshair test failed\n")
    
    # Test YOLO with crosshair
    print("2. Testing YOLO detection with crosshair...")
    if test_yolo_overlay():
        print("✓ YOLO overlay test completed")
    else:
        print("✗ YOLO overlay test failed")
    
    print("\nTest suite completed!")