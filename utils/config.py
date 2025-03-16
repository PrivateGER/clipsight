#!/usr/bin/env python3
"""
Configuration management for CLIP Image Search.
"""

import os
import json

class ConfigManager:
    def __init__(self, config_dir=None):
        """Initialize the config manager"""
        if config_dir is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".clip_search")
        
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, "config.json")
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, config):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
        
        return None 