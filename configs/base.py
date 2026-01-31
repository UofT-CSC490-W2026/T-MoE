from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json
from pathlib import Path
import torch


@dataclass
class BaseConfig:
    """Base configuration with serialization support."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize config to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            path.write_text(json_str)
        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, path: Path) -> "BaseConfig":
        """Load config from JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def update(self, **kwargs) -> "BaseConfig":
        """Return new config with updated values."""
        data = self.to_dict()
        data.update(kwargs)
        return self.__class__(**data)


@dataclass
class DeviceConfig(BaseConfig):
    """Device configuration."""

    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    dtype: str = "float32"  # "float32", "float16", "bfloat16"
    compile: bool = False  # Use torch.compile

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_dtype(self) -> torch.dtype:
        """Get torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.dtype not in dtype_map:
            raise ValueError(
                f"Invalid dtype: {self.dtype!r}. Valid options are: {list(dtype_map.keys())}"
            )
        return dtype_map[self.dtype]
