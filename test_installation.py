"""
Test script to verify installation
Run this after installing requirements to check if everything works
"""
import sys
print("Testing Bear Detection App Installation...")
print("-" * 50)

# Test 1: Python version
print("\n1. Testing Python version...")
print(f"   Python {sys.version}")
if sys.version_info >= (3, 9):
    print("   ✅ Python version OK")
else:
    print("   ❌ Python 3.9+ required")

# Test 2: Import core packages
print("\n2. Testing core packages...")
packages = {
    'cv2': 'OpenCV',
    'torch': 'PyTorch', 
    'ultralytics': 'Ultralytics YOLO',
    'PyQt6': 'PyQt6 GUI',
    'yaml': 'PyYAML',
    'numpy': 'NumPy'
}

all_good = True
for module, name in packages.items():
    try:
        __import__(module)
        print(f"   ✅ {name} installed")
    except ImportError:
        print(f"   ❌ {name} not found")
        all_good = False

# Test 3: Check CUDA/GPU
print("\n3. Testing GPU support...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("   ℹ️  No CUDA GPU found (CPU will be used)")
except:
    print("   ⚠️  Cannot check GPU")

# Test 4: Test camera
print("\n4. Testing camera...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("   ✅ Camera working")
        else:
            print("   ⚠️  Camera opened but couldn't read frame")
    else:
        print("   ❌ Cannot open camera")
except Exception as e:
    print(f"   ❌ Camera error: {e}")

# Test 5: Test YOLO model loading
print("\n5. Testing YOLO model...")
try:
    from ultralytics import YOLO
    print("   Attempting to load yolov8n.pt (smallest model)...")
    model = YOLO('yolov8n.pt')
    print("   ✅ YOLO model loads successfully")
except Exception as e:
    print(f"   ❌ Model loading error: {e}")

# Test 6: Test PyQt6
print("\n6. Testing GUI framework...")
try:
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    print("   ✅ PyQt6 GUI framework OK")
    app.quit()
except Exception as e:
    print(f"   ❌ GUI error: {e}")

# Summary
print("\n" + "=" * 50)
if all_good:
    print("✅ All tests passed! You can run: python main.py")
else:
    print("❌ Some tests failed. Run: pip install -r requirements.txt")
print("=" * 50)

input("\nPress Enter to exit...")