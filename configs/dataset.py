from dataclasses import dataclass
from typing import Dict, Any, Optional

from configs import BaseConfig
from catalog.dataset_catalog import (
    get_dataset_info as get_catalog_info,
    get_available_datasets,
    validate_dataset_key,
)


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for dataset loading and processing.

    Supports both predefined datasets from the catalog and custom datasets.
    When custom_dataset_name is provided, it takes precedence over dataset_key.

    Example:
        # Using a predefined dataset
        config = DatasetConfig(dataset_key="wikitext-2")

        # Using a custom dataset
        config = DatasetConfig(
            custom_dataset_name="my/dataset",
            custom_dataset_config="subset",
            text_column="content"
        )
    """

    # Dataset selection
    dataset_key: str = "wikitext-2"

    # Custom dataset (overrides dataset_key if provided)
    custom_dataset_name: Optional[str] = None
    custom_dataset_config: Optional[str] = None
    text_column: str = "text"

    # Data processing
    max_seq_len: int = 512
    num_samples: Optional[int] = None
    streaming: bool = False

    # Data split
    train_split: str = "train"
    eval_split: str = "validation"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.custom_dataset_name:
            if not validate_dataset_key(self.dataset_key):
                available = ", ".join(get_available_datasets())
                raise ValueError(
                    f"Invalid dataset_key '{self.dataset_key}'. "
                    f"Available: {available}"
                )

    @property
    def is_custom_dataset(self) -> bool:
        """Check if using a custom dataset."""
        return self.custom_dataset_name is not None

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get resolved dataset configuration.

        Returns:
            Dictionary with dataset configuration including:
            - name: HuggingFace dataset name
            - config: Dataset configuration/subset
            - text_column: Column containing text data
            - streaming: Whether to use streaming mode
        """
        if self.is_custom_dataset:
            return {
                "name": self.custom_dataset_name,
                "config": self.custom_dataset_config,
                "text_column": self.text_column,
                "streaming": self.streaming,
            }

        # Get from catalog and merge with instance overrides
        catalog_info = get_catalog_info(self.dataset_key).copy()

        # Allow instance-level overrides of streaming
        if self.streaming != catalog_info.get("streaming", False):
            catalog_info["streaming"] = self.streaming

        return catalog_info

    def get_description(self) -> str:
        if self.is_custom_dataset:
            return f"Custom dataset: {self.custom_dataset_name}"
        return get_catalog_info(self.dataset_key).get("description", self.dataset_key)
