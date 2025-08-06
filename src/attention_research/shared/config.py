"""Configuration management for attention research."""

import logging
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the attention research project."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize configuration.

        Args:
            config_path: Path to the configuration file. If None, uses default base.yaml.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "base.yaml"

        self.config_path = config_path
        self._config: DictConfig | None = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with self.config_path.open() as f:
                config_dict = yaml.safe_load(f)
            self._config = OmegaConf.create(config_dict)
        except FileNotFoundError as err:
            msg = f"Configuration file not found: {self.config_path}"
            raise FileNotFoundError(msg) from err
        except yaml.YAMLError as err:
            msg = f"Error parsing YAML configuration: {err}"
            raise ValueError(msg) from err

    @property
    def config(self) -> DictConfig:
        """Get the configuration object."""
        if self._config is None:
            self._load_config()
        if self._config is None:
            msg = "Configuration failed to load"
            raise RuntimeError(msg)
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Configuration key using dot notation (e.g., 'model.hidden_size')
            default: Default value if key not found

        Returns
        -------
            Configuration value
        """
        return OmegaConf.select(self.config, key, default=default)

    def update(self, updates: dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
        """
        if self._config is None:
            self._load_config()
        if self._config is None:
            msg = "Configuration failed to load"
            raise RuntimeError(msg)
        merged = OmegaConf.merge(self._config, updates)
        if not isinstance(merged, DictConfig):
            msg = "Configuration merge resulted in unexpected type"
            raise TypeError(msg)
        self._config = merged

    def set_framework(self, framework: str) -> None:
        """Set the framework (torch or jax).

        Args:
            framework: Framework name ('torch' or 'jax')
        """
        if framework not in ["torch", "jax"]:
            msg = f"Framework must be 'torch' or 'jax', got {framework}"
            raise ValueError(msg)
        self.update({"framework": framework})

    def set_attention_type(self, attention_type: str) -> None:
        """Set the attention mechanism type.

        Args:
            attention_type: Type of attention mechanism
        """
        valid_types = ["vanilla", "multi_head", "flash", "linear", "performer", "longformer"]
        if attention_type not in valid_types:
            msg = f"Attention type must be one of {valid_types}, got {attention_type}"
            raise ValueError(msg)
        self.update({"model.attention_type": attention_type})

    def get_framework(self) -> str:
        """Get the current framework."""
        return self.get("framework", "torch")

    def get_attention_type(self) -> str:
        """Get the current attention type."""
        return self.get("model.attention_type", "vanilla")

    def get_model_config(self) -> DictConfig:
        """Get model configuration."""
        model_config = self.get("model", {})
        return OmegaConf.create(model_config) if not isinstance(model_config, DictConfig) else model_config

    def get_training_config(self) -> DictConfig:
        """Get training configuration."""
        training_config = self.get("training", {})
        return OmegaConf.create(training_config) if not isinstance(training_config, DictConfig) else training_config

    def get_dataset_config(self) -> DictConfig:
        """Get dataset configuration."""
        dataset_config = self.get("dataset", {})
        return OmegaConf.create(dataset_config) if not isinstance(dataset_config, DictConfig) else dataset_config

    def save(self, path: Path | None = None) -> None:
        """Save current configuration to file.

        Args:
            path: Path to save configuration. If None, overwrites original file.
        """
        if path is None:
            path = self.config_path

        with path.open("w") as f:
            yaml.dump(OmegaConf.to_yaml(self._config), f, default_flow_style=False)

    def __repr__(self) -> str:
        """Return string representation of the configuration."""
        return f"Config(framework={self.get_framework()}, attention_type={self.get_attention_type()})"


def setup_logging(config: Config) -> None:
    """Set up logging configuration.

    Args:
        config: Configuration object
    """
    log_level = config.get("logging.level", "INFO")
    log_dir = Path(config.get("logging.log_dir", "./logs"))
    log_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "attention_research.log"), logging.StreamHandler()],
    )


# Global configuration instance
_global_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
