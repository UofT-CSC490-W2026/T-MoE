from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


# Single source of truth for all supported datasets.
# `splits`: maps logical split names to HuggingFace split names; val=None means
#           no validation split exists (prepare_data skips it).
# `streaming`: required for datasets > ~1B tokens (avoids loading into RAM).
# To add a dataset: add one entry here.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wikitext-2": {
        "hf_path": "wikitext",
        "hf_name": "wikitext-2-raw-v1",
        "text_column": "text",
        "splits": {"train": "train", "val": "validation"},
        "streaming": False,
    },
    "wikitext-103": {
        "hf_path": "wikitext",
        "hf_name": "wikitext-103-raw-v1",
        "text_column": "text",
        "splits": {"train": "train", "val": "validation"},
        "streaming": False,
    },
    # ~100 shards x 200MB = ~20GB. Recommended for 125M-1.3B runs.
    # streaming=False: HF downloads parquet files to cache, then parallel tokenization
    # with num_proc workers fires (~35 min on Modal vs ~25h sequential streaming).
    # Set HF_DATASETS_CACHE to the Modal Volume so the download persists.
    "fineweb-edu": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_name": "sample-10BT",
        "text_column": "text",
        "splits": {"train": "train", "val": None},
        "streaming": False,
    },
    # Full FineWeb-Edu (1.3T tokens) — production scale only. Keep streaming.
    "fineweb-edu-full": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_name": None,
        "text_column": "text",
        "splits": {"train": "train", "val": None},
        "streaming": True,
    },
    "openwebtext": {
        "hf_path": "openwebtext",
        "hf_name": None,
        "text_column": "text",
        "splits": {"train": "train", "val": None},
        "streaming": True,
    },
    "c4": {
        "hf_path": "allenai/c4",
        "hf_name": "en",
        "text_column": "text",
        "splits": {"train": "train", "val": "validation"},
        "streaming": True,
    },
}


def get_shard_dir(
    dataset_key: str, model_key: str, base: str = "data/shards"
) -> "Path":
    """
    Canonical shard directory: <base>/<dataset_key>/vocab<vocab_size>/

    Encoding vocab size in the path ensures that switching to a model with a
    different tokenizer (e.g. GPT-2 50257 → Llama 32000) automatically uses
    separate shard trees rather than silently loading wrong tokens.
    Models sharing a tokenizer (gpt-neo-125m and gpt-neo-1.3b both use GPT-2)
    resolve to the same directory and reuse shards without re-tokenizing.
    """
    from src.configs.model import model_lookup

    vocab_size = model_lookup(model_key)["tokenizer_vocab_size"]
    return Path(base) / dataset_key / f"vocab{vocab_size}"


def get_dataset_info(dataset_key: str) -> Dict[str, Any]:
    if dataset_key not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset key: '{dataset_key}'. "
            f"Available: {available}\n"
            f"To add a new dataset, edit DATASET_REGISTRY in src/configs/dataset.py."
        )
    return DATASET_REGISTRY[dataset_key]


@dataclass
class DatasetConfig:
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
        if self.custom_dataset_name:
            return {
                "hf_path": self.custom_dataset_name,
                "hf_name": self.custom_dataset_config,
                "text_column": self.text_column,
                "streaming": self.streaming,
            }
        info = dict(get_dataset_info(self.dataset_key))
        info["text_column"] = self.text_column
        return info

    def get_description(self) -> str:
        if self.custom_dataset_name:
            name = self.custom_dataset_name
        else:
            info = self.get_dataset_info()
            name = info["hf_path"]
            if info.get("hf_name"):
                name += f"/{info['hf_name']}"
        return f"{name} (seq_len={self.max_seq_len})"
