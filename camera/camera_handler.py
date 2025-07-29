"""
Camera handling module for Bear Detection Application
"""
import cv2
import numpy as np
from queue import Queue
from threading import Thread, Lock, Event
import time
from typing import Optional, Tuple
import logging


class CameraHandler:
    """Handles camera capture with threading and buffering"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Camera properties
        self.source = config.get('source', 0)
        self.resolution = config.get('resolution', {'width': 1280, 'height': 720})
        self.fps = config.get('fps', 30)
        self.buffer_size = config.get('buffer_size', 128)
        
        # Video capture object
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Threading components
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.capture_thread: Optional[Thread] = None
        self.is_running = Event()
        self.lock = Lock()
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.actual_fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
    
    def start(self) -> bool:
        """Start camera capture"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera source: {self.source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Start capture thread
            self.is_running.set()
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture"""
        self.is_running.clear()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        
        self.logger.info(f"Camera stopped. Total frames: {self.frame_count}, "
                        f"Dropped: {self.dropped_frames}")
    
    def _capture_loop(self):
        """Background thread for capturing frames"""
        while self.is_running.is_set():
            with self.lock:
                if not self.cap or not self.cap.isOpened():
                    break
                
                ret, frame = self.cap.read()
            
            if not ret:
                self.logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Update statistics
            self.frame_count += 1
            self._update_fps()
            
            # Try to add frame to queue
            try:
                if self.frame_queue.full():
                    # Remove old frame to make room
                    self.frame_queue.get_nowait()
                    self.dropped_frames += 1
                
                self.frame_queue.put(frame, block=False)
            except:
                self.dropped_frames += 1
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the latest frame from the camera"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.actual_fps = self.fps_frame_count / (current_time - self.last_fps_time)
            self.fps_frame_count = 0
            self.last_fps_time = current_time
    
    def get_stats(self) -> dict:
        """Get camera statistics"""
        return {
            'fps': self.actual_fps,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'buffer_usage': self.frame_queue.qsize() / self.buffer_size
        }
    
    def is_active(self) -> bool:
        """Check if camera is active"""
        return self.is_running.is_set() and self.cap is not None
    
    def take_snapshot(self, path: str) -> bool:
        """Save current frame to file"""
        frame = self.get_frame()
        if frame is not None:
            try:
                cv2.imwrite(path, frame)
                self.logger.info(f"Snapshot saved to: {path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save snapshot: {e}")
        return False