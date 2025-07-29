"""
YOLO-based bear detector module
"""
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import logging
from threading import Lock
import time


class BearDetector:
    """YOLO-based detector specifically configured for bear detection"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Model settings
        self.model_name = config.get('name', 'yolov8s.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.device = self._get_device(config.get('device', 'auto'))
        
        # Model instance and lock for thread safety
        self.model: Optional[YOLO] = None
        self.model_lock = Lock()
        
        # Performance tracking
        self.inference_times = []
        self.max_inference_history = 100
        
        # Class names we're interested in
        self.target_classes = ['bear']  # Will be updated after model loads
    
    def _get_device(self, device_config: str) -> str:
        """Determine the best device to use"""
        if device_config == 'auto':
            if torch.cuda.is_available():
                self.logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
                return 'cuda:0'
            else:
                self.logger.info("CUDA not available. Using CPU")
                return 'cpu'
        return device_config
    
    def load_model(self) -> bool:
        """Load the YOLO model"""
        try:
            self.logger.info(f"Loading YOLO model: {self.model_name}")
            
            with self.model_lock:
                self.model = YOLO(self.model_name)
                
                # Get model info
                if hasattr(self.model.model, 'names'):
                    self.logger.info(f"Model classes: {self.model.model.names}")
                    
                    # Find bear class ID
                    for class_id, class_name in self.model.model.names.items():
                        if 'bear' in class_name.lower():
                            self.target_classes = [class_name]
                            self.logger.info(f"Found bear class: '{class_name}' (ID: {class_id})")
                            break
                
                # Move model to device
                self.model.to(self.device)
                
            self.logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform bear detection on a frame
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - class_name: str
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return []
        
        try:
            start_time = time.time()
            
            with self.model_lock:
                # Run inference
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )
            
            # Process results
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = self.model.model.names[class_id]
                        
                        # Check if this is a bear detection
                        if 'bear' in class_name.lower() or class_name in self.target_classes:
                            detection = {
                                'bbox': box.xyxy[0].cpu().numpy().astype(int),
                                'confidence': float(box.conf),
                                'class_name': class_name,
                                'class_id': class_id
                            }
                            detections.append(detection)
            
            # Track inference time
            inference_time = time.time() - start_time
            self._track_inference_time(inference_time)
            
            if detections:
                self.logger.debug(f"Detected {len(detections)} bear(s) in {inference_time:.3f}s")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            return []
    
    def _track_inference_time(self, time_seconds: float):
        """Track inference times for performance monitoring"""
        self.inference_times.append(time_seconds)
        if len(self.inference_times) > self.max_inference_history:
            self.inference_times.pop(0)
    
    def get_stats(self) -> dict:
        """Get detector statistics"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0,
                'min_inference_time': 0,
                'max_inference_time': 0,
                'inference_fps': 0
            }
        
        avg_time = np.mean(self.inference_times)
        return {
            'avg_inference_time': avg_time,
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'inference_fps': 1.0 / avg_time if avg_time > 0 else 0
        }
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        import cv2
        
        # Make a copy to avoid modifying original
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for bear
            thickness = 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA
            )
        
        return annotated_frame
    
    def update_threshold(self, confidence: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, confidence))
        self.logger.info(f"Confidence threshold updated to: {self.confidence_threshold}")
    
    def cleanup(self):
        """Cleanup resources"""
        with self.model_lock:
            self.model = None
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA