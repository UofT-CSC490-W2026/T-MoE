from typing import Dict, Any

# Dataset catalog - maps short keys to HuggingFace dataset configurations
DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "wikitext-2": {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "text_column": "text",
        "streaming": False,
        "description": "WikiText-2 (small, ~2M tokens)",
    },
    "wikitext-103": {
        "name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "text_column": "text",
        "streaming": False,
        "description": "WikiText-103 (large, ~103M tokens)",
    },
    "c4": {
        "name": "allenai/c4",
        "config": "en",
        "text_column": "text",
        "streaming": True,
        "description": "C4 (Colossal Clean Crawled Corpus)",
    },
    "openwebtext": {
        "name": "openwebtext",
        "config": None,
        "text_column": "text",
        "streaming": False,
        "description": "OpenWebText (Reddit outlinks)",
    },
    "the_pile": {
        "name": "monology/pile-uncopyrighted",
        "config": None,
        "text_column": "text",
        "streaming": True,
        "description": "The Pile (diverse text corpus)",
    },
    "code": {
        "name": "codeparrot/github-code",
        "config": "all-all",
        "text_column": "code",
        "streaming": True,
        "description": "GitHub Code (for code domain)",
    },
}


def get_available_datasets() -> list[str]:
    """Return list of available dataset keys."""
    return list(DATASET_CATALOG.keys())


def get_dataset_info(dataset_key: str) -> Dict[str, Any]:
    """Get dataset configuration by key."""
    if dataset_key not in DATASET_CATALOG:
        available = ", ".join(get_available_datasets())
        raise KeyError(
            f"Dataset '{dataset_key}' not found in catalog. "
            f"Available datasets: {available}"
        )
    return DATASET_CATALOG[dataset_key]


def validate_dataset_key(dataset_key: str) -> bool:
    """Check if a dataset key exists in the catalog."""
    return dataset_key in DATASET_CATALOG
