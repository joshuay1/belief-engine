import yaml
import os
import argparse
import copy
from typing import Dict, Any, Optional

# Resolve defaults.yaml relative to the project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULTS_PATH = os.path.join(_PROJECT_ROOT, "config", "defaults.yaml")

class ConfigLoader:
    """
    Centralized configuration loader.
    Loads config/defaults.yaml as the base layer, then deep-merges
    the user's config on top. Used by all experiment types.
    """
    
    _defaults_cache = None

    @classmethod
    def _load_defaults(cls) -> Dict[str, Any]:
        """Loads and caches config/defaults.yaml."""
        if cls._defaults_cache is None:
            if os.path.exists(_DEFAULTS_PATH):
                with open(_DEFAULTS_PATH, 'r') as f:
                    cls._defaults_cache = yaml.safe_load(f) or {}
            else:
                cls._defaults_cache = {}
        return cls._defaults_cache

    @staticmethod
    def load_config(config_path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Loads a YAML config file, merges it with defaults.yaml, and applies overrides.
        
        Args:
            config_path (str): Path to the YAML file.
            overrides (Dict): Optional dictionary of overrides (e.g. from CLI).
            
        Returns:
            Dict: The final configuration dictionary.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            try:
                user_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML: {e}")

        # Start with defaults.yaml as base layer
        final_config = copy.deepcopy(ConfigLoader._load_defaults())
        
        # Deep merge user config on top
        ConfigLoader._deep_update(final_config, user_config)
        
        # Apply overrides
        if overrides:
            ConfigLoader._deep_update(final_config, overrides)
            
        return final_config

    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict):
        """Recursively updates base_dict with update_dict."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                ConfigLoader._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
                
    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """
        Validates the configuration against required schema rules.
        Returns True if valid, raises ValueError otherwise.
        """
        # Example validation rules
        if "topic" not in config or not config["topic"]:
            raise ValueError("Config missing required field: 'topic'")
            
        return True
