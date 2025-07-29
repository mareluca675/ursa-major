"""
Bear Detection Desktop Application
Main entry point
"""
import sys
import os
import argparse
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.logger import setup_logger, DetectionLogger
from camera.camera_handler import CameraHandler
from models.detector import BearDetector
from gui.main_window import MainWindow


class BearDetectionApp:
    """Main application class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.logger = setup_logger("BearDetection", self.config.logging)
        self.detection_logger = None
        
        # Initialize components
        self.camera_handler = None
        self.detector = None
        self.main_window = None
        
        self.logger.info("Bear Detection Application starting...")
    
    def initialize_components(self):
        """Initialize all application components"""
        try:
            # Initialize detection logger if enabled
            if self.config.detection.get('save_detections', False):
                log_path = self.config.detection.get('detection_log_path', './logs/detections.log')
                self.detection_logger = DetectionLogger(log_path)
            
            # Initialize camera
            self.logger.info("Initializing camera...")
            self.camera_handler = CameraHandler(self.config.camera, self.logger)
            if not self.camera_handler.start():
                raise RuntimeError("Failed to initialize camera")
            
            # Initialize detector
            self.logger.info("Initializing bear detector...")
            self.detector = BearDetector(self.config.model, self.logger)
            if not self.detector.load_model():
                raise RuntimeError("Failed to load detection model")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            return False
    
    def run(self):
        """Run the application"""
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("Bear Detection System")
        
        # Set application style
        app.setStyle('Fusion')
        
        # Initialize components
        if not self.initialize_components():
            QMessageBox.critical(
                None,
                "Initialization Error",
                "Failed to initialize application components.\n"
                "Please check the logs for details."
            )
            return 1
        
        try:
            # Create and show main window
            self.main_window = MainWindow(
                self.camera_handler,
                self.detector,
                self.config,
                self.logger
            )
            self.main_window.show()
            
            # Run application
            self.logger.info("Application started successfully")
            return app.exec()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            QMessageBox.critical(
                None,
                "Application Error",
                f"An error occurred: {str(e)}"
            )
            return 1
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        if self.camera_handler:
            self.camera_handler.stop()
        
        if self.detector:
            self.detector.cleanup()
        
        self.logger.info("Application shutdown complete")


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Bear Detection Desktop Application")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index (overrides config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = BearDetectionApp(args.config)
    
    # Apply command line overrides
    if args.camera is not None:
        app.config.set('camera.source', args.camera)
    if args.model is not None:
        app.config.set('model.name', args.model)
    if args.confidence is not None:
        app.config.set('model.confidence_threshold', args.confidence)
    
    # Run application
    sys.exit(app.run())


if __name__ == "__main__":
    main()