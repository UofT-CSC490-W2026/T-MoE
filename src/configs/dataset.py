from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


# Dataset catalog: maps dataset_key to HuggingFace load_dataset arguments
DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "wikitext-2": {
        "hf_path": "wikitext",
        "hf_name": "wikitext-2-raw-v1",
        "text_column": "text",
        "streaming": False,
    },
    "wikitext-103": {
        "hf_path": "wikitext",
        "hf_name": "wikitext-103-raw-v1",
        "text_column": "text",
        "streaming": False,
    },
    "openwebtext": {
        "hf_path": "openwebtext",
        "hf_name": None,
        "text_column": "text",
        "streaming": True,
    },
    "c4": {
        "hf_path": "allenai/c4",
        "hf_name": "en",
        "text_column": "text",
        "streaming": True,
    },
}


@dataclass
class DatasetConfig:
    """
    Dataset configuration dataclass used by experiment.py.

    Provides a typed interface over the dataset YAML config keys and exposes
    catalog lookup methods for obtaining HuggingFace dataset identifiers.
    """

    dataset_key: str = "wikitext-2"
    custom_dataset_name: Optional[str] = None
    custom_dataset_config: Optional[str] = None
    text_column: str = "text"
    max_seq_len: int = 512
    num_samples: Optional[int] = None
    streaming: bool = False
    train_split: str = "train"
    eval_split: Optional[str] = "validation"

    def get_dataset_info(self) -> Dict[str, Any]:
        """Look up dataset metadata from the catalog."""
        if self.custom_dataset_name:
            return {
                "hf_path": self.custom_dataset_name,
                "hf_name": self.custom_dataset_config,
                "text_column": self.text_column,
                "streaming": self.streaming,
            }

        if self.dataset_key not in DATASET_CATALOG:
            available = list(DATASET_CATALOG.keys())
            raise ValueError(
                f"Unknown dataset key: '{self.dataset_key}'. "
                f"Available: {available}\n"
                f"To add a new dataset, edit DATASET_CATALOG in src/configs/dataset.py "
                f"and DATASET_REGISTRY in scripts/prepare_data.py."
            )

        info = dict(DATASET_CATALOG[self.dataset_key])
        # Allow text_column override from config
        info["text_column"] = self.text_column
        return info

    def get_description(self) -> str:
        """Human-readable dataset description."""
        if self.custom_dataset_name:
            name = self.custom_dataset_name
        else:
            info = self.get_dataset_info()
            name = info["hf_path"]
            if info.get("hf_name"):
                name += f"/{info['hf_name']}"
        return f"{name} (seq_len={self.max_seq_len})"
