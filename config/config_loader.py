"""
Configuration loader for the fraud detection pipeline.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration for the fraud detection pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration loader."""
        self.config_path = config_path or "config/pipeline_config.yaml"
        self.config = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )

            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)

            # Override with environment variables
            self._override_with_env()

            return self.config

        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    def _override_with_env(self):
        """Override configuration with environment variables."""
        env_mappings = {
            "KAFKA_BOOTSTRAP_SERVERS": ("kafka", "bootstrap_servers"),
            "KAFKA_TOPIC_TRANSACTIONS": ("kafka", "topic_transactions"),
            "REDIS_HOST": ("redis", "host"),
            "REDIS_PORT": ("redis", "port"),
            "MODEL_PATH": ("model", "path"),
            "RISK_THRESHOLD_HIGH": ("risk_thresholds", "high"),
            "DASHBOARD_PORT": ("dashboard", "port"),
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(self.config, config_path, env_value)

    def _set_nested_value(self, config: Dict[str, Any], path: tuple, value: Any):
        """Set a nested value in the configuration dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert value type if needed
        if isinstance(value, str):
            # Try to convert to appropriate type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)

        current[path[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration."""
        return self.config.get("kafka", {})

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return self.config.get("redis", {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get ML model configuration."""
        return self.config.get("model", {})

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.config.get("processing", {})

    def get_alert_config(self) -> Dict[str, Any]:
        """Get alert configuration."""
        return self.config.get("alerts", {})

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.config.get("dashboard", {})

    def get_simulator_config(self) -> Dict[str, Any]:
        """Get data simulator configuration."""
        return self.config.get("simulator", {})

    def get_risk_thresholds(self) -> Dict[str, float]:
        """Get risk thresholds."""
        return self.config.get("risk_thresholds", {})

    def get_performance_targets(self) -> Dict[str, Any]:
        """Get performance targets."""
        return self.config.get("performance", {})

    def validate_config(self) -> bool:
        """Validate the configuration."""
        required_sections = ["kafka", "redis", "model", "processing", "alerts"]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate specific values
        if self.get("kafka.bootstrap_servers") is None:
            raise ValueError("Kafka bootstrap servers not configured")

        if self.get("redis.host") is None:
            raise ValueError("Redis host not configured")

        if self.get("model.path") is None:
            raise ValueError("Model path not configured")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary."""
        return self.config.copy()

    def save_config(self, path: Optional[str] = None) -> bool:
        """Save the current configuration to a file."""
        try:
            save_path = path or self.config_path
            with open(save_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            raise RuntimeError(f"Error saving configuration: {e}")


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Convenience function to load configuration."""
    return ConfigLoader(config_path)


# Example usage
if __name__ == "__main__":
    config = load_config()
    print("Kafka config:", config.get_kafka_config())
    print("Redis config:", config.get_redis_config())
    print("Model config:", config.get_model_config())
