# Bear Detection Desktop Application

A real-time bear detection system using YOLOv8 and PyQt6, designed for wildlife monitoring and safety applications.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- 🐻 Real-time bear detection using state-of-the-art YOLO models
- 📹 Live webcam feed with bounding box visualization
- 🎛️ Adjustable confidence threshold
- 📊 Performance statistics and detection logging
- 🎨 Modern dark-themed GUI
- 🔧 Modular architecture for easy Arduino integration
- ⚡ Multi-threaded for smooth performance

## System Requirements

- **OS**: Windows 10/11 (Linux and macOS compatible with minor adjustments)
- **Python**: 3.10 or 3.11 (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Webcam**: Any USB webcam (tested with Microsoft LifeCam)

## Installation

### 1. Clone or Download the Project

```bash
# Create a project directory
mkdir bear-detection
cd bear-detection

# Copy all the provided files into this directory
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. GPU Support (Optional but Recommended)

For NVIDIA GPU acceleration:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
bear-detection/
├── main.py              # Main entry point
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── camera/             # Camera handling module
│   ├── __init__.py
│   └── camera_handler.py
├── models/             # Detection models
│   ├── __init__.py
│   └── detector.py
├── gui/                # User interface
│   ├── __init__.py
│   └── main_window.py
├── utils/              # Utility modules
│   ├── __init__.py
│   ├── config.py
│   └── logger.py
└── logs/               # Log files (created automatically)
```

## Usage

### Basic Usage

1. **Start the application**:
   ```bash
   python main.py
   ```

2. **The GUI will open showing**:
   - Live camera feed on the left
   - Detection status indicator
   - Control panel on the right
   - Statistics and detection log

3. **Controls**:
   - **Stop/Start Detection**: Pause/resume bear detection
   - **Take Snapshot**: Save current frame as image
   - **Confidence Slider**: Adjust detection sensitivity (0-100%)
   - **Show Bounding Boxes**: Toggle detection visualization
   - **Show FPS**: Toggle FPS overlay

### Command Line Options

```bash
# Use a different camera
python main.py --camera 1

# Use a different model
python main.py --model yolov8m.pt

# Set confidence threshold
python main.py --confidence 0.7

# Use custom config file
python main.py --config my_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

### Model Settings
- `model.name`: YOLOv8 variant (yolov8n/s/m/l/x.pt)
- `model.confidence_threshold`: Detection confidence (0.0-1.0)
- `model.device`: "auto", "cpu", or "cuda:0"

### Camera Settings
- `camera.source`: Camera index (0 for default)
- `camera.resolution`: Video resolution
- `camera.fps`: Target frame rate

### GUI Settings
- `gui.theme`: "dark" or "light"
- `gui.window_size`: Initial window dimensions

## Troubleshooting

### Camera Issues

**Problem**: "Failed to initialize camera"
- **Solution**: Check if another application is using the camera
- Try a different camera index: `python main.py --camera 1`

**Problem**: Low FPS or laggy video
- **Solution**: Lower the resolution in `config.yaml`
- Reduce `camera.fps` to 15 or 20

### Model Issues

**Problem**: "Failed to load detection model"
- **Solution**: The model will download automatically on first run
- Ensure you have internet connection
- Check disk space (models are ~25-150MB)

**Problem**: No bears detected
- **Solution**: YOLO models are trained on COCO dataset where bears might have limited representation
- Lower the confidence threshold using the slider
- Consider training a custom model on bear-specific datasets

### Performance Issues

**Problem**: High CPU usage
- **Solution**: Use GPU acceleration (see installation)
- Use lighter model: change to `yolov8n.pt` in config
- Reduce camera resolution

**Problem**: High memory usage
- **Solution**: Reduce `camera.buffer_size` in config
- Close other applications

## Future Arduino Integration

The application is designed with hardware integration in mind:

1. **Serial Communication**: Already includes `pyserial` dependency
2. **Modular Architecture**: Easy to add hardware controller
3. **Event-Driven**: Detection events can trigger hardware alerts

To add Arduino support:
1. Set `hardware.enabled: true` in config.yaml
2. Configure serial port settings
3. Implement alert logic in Arduino sketch

## Model Information

The application uses YOLOv8 models:
- **yolov8n**: Nano (fastest, least accurate)
- **yolov8s**: Small (balanced) - **Default**
- **yolov8m**: Medium (slower, more accurate)
- **yolov8l**: Large (slowest, most accurate)

Models are downloaded automatically from Ultralytics on first use.

## Development

### Adding Custom Features

The modular architecture makes it easy to extend:

1. **Custom Detectors**: Implement new detection algorithms in `models/`
2. **GUI Enhancements**: Add widgets in `gui/main_window.py`
3. **Hardware Support**: Add controllers in a new `hardware/` module

### Logging

- Application logs: `logs/app.log`
- Detection events: `logs/detections.log` (if enabled)

## Known Limitations

1. **Bear Detection Accuracy**: COCO dataset has limited bear samples
2. **Windows Defender**: May flag first run (false positive from PyInstaller)
3. **USB Camera Support**: Some cameras may require specific drivers

## Tips for Best Results

1. **Lighting**: Ensure good lighting conditions
2. **Camera Position**: Mount camera at appropriate height
3. **Background**: Minimize clutter in camera view
4. **Testing**: Use bear images/videos on screen for testing

## Credits

- **YOLO**: Ultralytics YOLOv8
- **GUI Framework**: PyQt6
- **Computer Vision**: OpenCV
- **Deep Learning**: PyTorch

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in the `logs/` directory
3. Ensure all dependencies are correctly installed
4. Try with different model variants or settings