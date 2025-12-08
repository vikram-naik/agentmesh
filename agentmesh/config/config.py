import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path

class AgentMeshConfig:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentMeshConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # 1. Load defaults
        # Assuming default.yaml is in the same directory as this file
        base_path = Path(__file__).parent
        default_config_path = base_path / "default.yaml"
        
        if default_config_path.exists():
            with open(default_config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

        # 2. Load override from env var if present
        override_path = os.getenv("AGENTMESH_CONFIG_PATH")
        if override_path and os.path.exists(override_path):
            with open(override_path, "r") as f:
                override_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config, override_config)

    def _deep_merge(self, base: Dict, override: Dict):
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)"""
        cls._instance = None
