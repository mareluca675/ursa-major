"""
Setup script for Bear Detection Application
Checks system requirements and helps with initial setup
"""
import sys
import subprocess
import os
import platform


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 9:
        print("❌ Python 3.9+ is required")
        return False
    
    if version.minor > 11:
        print("⚠️  Python 3.12+ may have compatibility issues with some packages")
    
    print("✅ Python version is compatible")
    return True


def check_pip():
    """Check if pip is available"""
    try:
        import pip
        print("✅ pip is installed")
        return True
    except ImportError:
        print("❌ pip is not installed")
        return False


def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            print("✅ Camera detected")
            return True
        else:
            print("⚠️  No camera detected (app will still run for testing)")
            return True
    except:
        print("⚠️  Cannot check camera (OpenCV not installed yet)")
        return True


def check_gpu():
    """Check for NVIDIA GPU and CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("ℹ️  No CUDA GPU detected (CPU will be used)")
            return True
    except:
        print("ℹ️  PyTorch not installed yet")
        return True


def create_directories():
    """Create necessary directories"""
    dirs = ['logs', 'snapshots']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"ℹ️  Directory exists: {dir_name}")


def main():
    """Run setup checks"""
    print("Bear Detection Application Setup")
    print("=" * 50)
    print()
    
    # Check system
    print(f"Operating System: {platform.system()} {platform.release()}")
    print()
    
    # Run checks
    checks = [
        ("Checking Python version...", check_python_version),
        ("Checking pip...", check_pip),
        ("Creating directories...", create_directories),
    ]
    
    all_passed = True
    for description, check_func in checks:
        print(description)
        if not check_func():
            all_passed = False
        print()
    
    if all_passed:
        print("✅ All checks passed!")
        print()
        print("Next steps:")
        print("1. Create virtual environment: python -m venv venv")
        print("2. Activate it: venv\\Scripts\\activate (Windows)")
        print("3. Install requirements: pip install -r requirements.txt")
        print("4. Run the app: python main.py")
        print()
        print("Or simply run: run_app.bat")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()