"""
Logging configuration for Bear Detection Application
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(name: str, config: dict) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        config: Configuration dictionary with logging settings
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.get('file_path', './logs/app.log'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(config.get('level', 'INFO'))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        config.get('file_path', './logs/app.log'),
        maxBytes=config.get('max_file_size_mb', 10) * 1024 * 1024,
        backupCount=config.get('backup_count', 5)
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class DetectionLogger:
    """Special logger for bear detection events"""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def log_detection(self, confidence: float, bbox: tuple, frame_number: int):
        """Log a bear detection event"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(self.log_path, 'a') as f:
            f.write(f"{timestamp},BEAR_DETECTED,confidence:{confidence:.3f},"
                   f"bbox:{bbox},frame:{frame_number}\n")