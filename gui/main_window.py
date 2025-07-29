"""
Main GUI window for Bear Detection Application
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QGroupBox, QGridLayout, QCheckBox,
    QTextEdit, QSplitter, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
import time
from typing import Optional, List, Dict
import logging


class VideoThread(QThread):
    """Thread for processing video frames"""
    frame_ready = pyqtSignal(np.ndarray, list)  # frame, detections
    stats_ready = pyqtSignal(dict)  # statistics
    
    def __init__(self, camera_handler, detector, config):
        super().__init__()
        self.camera_handler = camera_handler
        self.detector = detector
        self.config = config
        self.is_running = True
        self.process_frames = True
        
    def run(self):
        """Main processing loop"""
        while self.is_running:
            if not self.process_frames:
                time.sleep(0.01)
                continue
                
            # Get frame from camera
            frame = self.camera_handler.get_frame(timeout=0.1)
            if frame is None:
                continue
            
            # Perform detection
            detections = []
            if self.config.get('detection', {}).get('enabled', True):
                detections = self.detector.detect(frame)
            
            # Emit results
            self.frame_ready.emit(frame, detections)
    
    def stop(self):
        """Stop the thread"""
        self.is_running = False
        self.wait()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, camera_handler, detector, config, logger):
        super().__init__()
        self.camera_handler = camera_handler
        self.detector = detector
        self.config = config
        self.logger = logger
        
        # GUI configuration
        self.gui_config = config.gui
        
        # Video processing thread
        self.video_thread: Optional[VideoThread] = None
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_time = None
        
        # Initialize UI
        self.init_ui()
        
        # Start processing
        self.start_processing()
        
        # Setup statistics timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_statistics)
        self.stats_timer.start(1000)  # Update every second
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(self.gui_config.get('window_title', 'Bear Detection System'))
        self.resize(
            self.gui_config.get('window_size', {}).get('width', 1200),
            self.gui_config.get('window_size', {}).get('height', 800)
        )
        
        # Set dark theme if requested
        if self.gui_config.get('theme', 'dark') == 'dark':
            self.setStyleSheet(self.get_dark_theme_style())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Video display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(False)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #555;")
        left_layout.addWidget(self.video_label)
        
        # Detection status
        self.status_label = QLabel("No Bear Detected")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.status_label.setMinimumHeight(60)
        self.update_status(False)
        left_layout.addWidget(self.status_label)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Controls and info
        right_panel = QWidget()
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        # Start/Stop button
        self.start_stop_button = QPushButton("Stop Detection")
        self.start_stop_button.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.start_stop_button)
        
        # Snapshot button
        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.clicked.connect(self.take_snapshot)
        controls_layout.addWidget(self.snapshot_button)
        
        # Confidence threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.config.model['confidence_threshold'] * 100))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel(f"{self.threshold_slider.value()}%")
        threshold_layout.addWidget(self.threshold_label)
        controls_layout.addLayout(threshold_layout)
        
        # Options checkboxes
        self.show_boxes_checkbox = QCheckBox("Show Bounding Boxes")
        self.show_boxes_checkbox.setChecked(self.gui_config.get('draw_boxes', True))
        controls_layout.addWidget(self.show_boxes_checkbox)
        
        self.show_fps_checkbox = QCheckBox("Show FPS")
        self.show_fps_checkbox.setChecked(self.gui_config.get('show_fps', True))
        controls_layout.addWidget(self.show_fps_checkbox)
        
        controls_group.setLayout(controls_layout)
        right_layout.addWidget(controls_group)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        
        # Statistics labels
        self.fps_label = QLabel("FPS: 0")
        self.frames_label = QLabel("Frames: 0")
        self.detections_label = QLabel("Detections: 0")
        self.inference_label = QLabel("Inference: 0ms")
        
        stats_layout.addWidget(QLabel("Camera FPS:"), 0, 0)
        stats_layout.addWidget(self.fps_label, 0, 1)
        stats_layout.addWidget(QLabel("Total Frames:"), 1, 0)
        stats_layout.addWidget(self.frames_label, 1, 1)
        stats_layout.addWidget(QLabel("Bears Detected:"), 2, 0)
        stats_layout.addWidget(self.detections_label, 2, 1)
        stats_layout.addWidget(QLabel("Inference Time:"), 3, 0)
        stats_layout.addWidget(self.inference_label, 3, 1)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # Log display
        log_group = QGroupBox("Detection Log")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(200)
        log_layout.addWidget(self.log_display)
        
        # Clear log button
        clear_log_button = QPushButton("Clear Log")
        clear_log_button.clicked.connect(self.log_display.clear)
        log_layout.addWidget(clear_log_button)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        right_layout.addStretch()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([800, 400])
    
    def start_processing(self):
        """Start video processing thread"""
        self.video_thread = VideoThread(
            self.camera_handler,
            self.detector,
            self.config
        )
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()
    
    @pyqtSlot(np.ndarray, list)
    def update_frame(self, frame: np.ndarray, detections: List[Dict]):
        """Update the video display with new frame"""
        self.frame_count += 1
        
        # Draw detections if enabled
        if self.show_boxes_checkbox.isChecked() and detections:
            frame = self.detector.draw_detections(frame, detections)
        
        # Add FPS overlay if enabled
        if self.show_fps_checkbox.isChecked():
            fps = self.camera_handler.get_stats()['fps']
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        
        # Convert frame to QPixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage
        q_image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update detection status
        if detections:
            self.detection_count += len(detections)
            self.last_detection_time = time.time()
            self.update_status(True)
            self.log_detection(detections)
        else:
            # Clear status if no detection for 2 seconds
            if (self.last_detection_time and 
                time.time() - self.last_detection_time > 2.0):
                self.update_status(False)
    
    def update_status(self, bear_detected: bool):
        """Update detection status display"""
        if bear_detected:
            self.status_label.setText("🐻 BEAR DETECTED! 🐻")
            self.status_label.setStyleSheet(
                "background-color: #ff4444; color: white; "
                "border-radius: 10px; padding: 10px;"
            )
        else:
            self.status_label.setText("No Bear Detected")
            self.status_label.setStyleSheet(
                "background-color: #44ff44; color: black; "
                "border-radius: 10px; padding: 10px;"
            )
    
    def log_detection(self, detections: List[Dict]):
        """Log detection event"""
        timestamp = time.strftime("%H:%M:%S")
        for detection in detections:
            confidence = detection['confidence']
            log_msg = f"[{timestamp}] Bear detected with {confidence:.1%} confidence\n"
            self.log_display.append(log_msg)
    
    def update_statistics(self):
        """Update statistics display"""
        # Camera stats
        camera_stats = self.camera_handler.get_stats()
        self.fps_label.setText(f"{camera_stats['fps']:.1f}")
        self.frames_label.setText(str(self.frame_count))
        self.detections_label.setText(str(self.detection_count))
        
        # Detector stats
        detector_stats = self.detector.get_stats()
        inference_ms = detector_stats['avg_inference_time'] * 1000
        self.inference_label.setText(f"{inference_ms:.1f}ms")
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        if self.video_thread:
            self.video_thread.process_frames = not self.video_thread.process_frames
            if self.video_thread.process_frames:
                self.start_stop_button.setText("Stop Detection")
            else:
                self.start_stop_button.setText("Start Detection")
    
    def on_threshold_changed(self, value: int):
        """Handle threshold slider change"""
        threshold = value / 100.0
        self.threshold_label.setText(f"{value}%")
        self.detector.update_threshold(threshold)
    
    def take_snapshot(self):
        """Take a snapshot of current frame"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"
        if self.camera_handler.take_snapshot(filename):
            QMessageBox.information(self, "Snapshot", f"Saved as {filename}")
    
    def get_dark_theme_style(self) -> str:
        """Get dark theme stylesheet"""
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            background-color: #3c3c3c;
            border: 2px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #4c4c4c;
            border: 1px solid #666;
            border-radius: 5px;
            padding: 5px;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color: #5c5c5c;
        }
        QPushButton:pressed {
            background-color: #3c3c3c;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #4c4c4c;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #888;
            border: 1px solid #666;
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }
        QTextEdit {
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 5px;
        }
        """
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.video_thread:
            self.video_thread.stop()
        event.accept()