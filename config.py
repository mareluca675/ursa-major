"""
Configuration management for Bear Detection Application
"""
import yaml
import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager with support for YAML files and environment variables"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Example: BEAR_DETECTOR_MODEL_CONFIDENCE_THRESHOLD=0.7
        prefix = "BEAR_DETECTOR_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert environment variable to config path
                config_path = key[len(prefix):].lower().split('_')
                self._set_nested(self._config, config_path, value)
    
    def _set_nested(self, d: dict, path: list, value: str):
        """Set a nested dictionary value from a path"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        
        # Try to convert value to appropriate type
        try:
            if value.lower() in ('true', 'false'):
                d[path[-1]] = value.lower() == 'true'
            elif '.' in value:
                d[path[-1]] = float(value)
            else:
                d[path[-1]] = int(value)
        except ValueError:
            d[path[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('model.confidence_threshold', 0.5)
        """
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = path.split('.')
        d = self._config
        
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        
        d[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    @property
    def model(self) -> dict:
        """Get model configuration"""
        return self._config.get('model', {})
    
    @property
    def camera(self) -> dict:
        """Get camera configuration"""
        return self._config.get('camera', {})
    
    @property
    def gui(self) -> dict:
        """Get GUI configuration"""
        return self._config.get('gui', {})
    
    @property
    def detection(self) -> dict:
        """Get detection configuration"""
        return self._config.get('detection', {})
    
    @property
    def hardware(self) -> dict:
        """Get hardware configuration"""
        return self._config.get('hardware', {})
    
    @property
    def logging(self) -> dict:
        """Get logging configuration"""
        return self._config.get('logging', {})
    
    @property
    def performance(self) -> dict:
        """Get performance configuration"""
        return self._config.get('performance', {})