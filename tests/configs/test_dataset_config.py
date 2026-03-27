import pytest

from pathlib import Path

from src.configs.dataset import (
    DATASET_REGISTRY,
    get_dataset_info,
    get_shard_dir,
    DatasetConfig,
)


class TestDatasetRegistry:
    def test_contains_expected_datasets(self):

        for key in ("wikitext-2", "wikitext-103", "fineweb-edu", "openwebtext", "c4"):
            assert key in DATASET_REGISTRY, f"Missing dataset: {key}"

    def test_each_entry_has_required_fields(self):

        required = {"hf_path", "text_column", "splits", "streaming"}

        for key, info in DATASET_REGISTRY.items():
            missing = required - set(info.keys())

            assert not missing, f"{key} missing fields: {missing}"

    def test_splits_have_train_key(self):

        for key, info in DATASET_REGISTRY.items():
            assert "train" in info["splits"], f"{key} has no train split"

    def test_streaming_datasets_have_no_val(self):

        for key, info in DATASET_REGISTRY.items():
            if info["streaming"]:
                val = info["splits"].get("val")

                assert val is None or isinstance(val, str)


class TestGetDatasetInfo:
    def test_returns_correct_info(self):

        info = get_dataset_info("wikitext-2")

        assert info["hf_path"] == "wikitext"

        assert info["hf_name"] == "wikitext-2-raw-v1"

        assert info["streaming"] is False

    def test_raises_for_unknown_key(self):

        with pytest.raises(ValueError, match="Unknown dataset key"):
            get_dataset_info("nonexistent-dataset")

    def test_error_lists_available_datasets(self):

        with pytest.raises(ValueError) as exc:
            get_dataset_info("bad-key")

        assert "wikitext-2" in str(exc.value)


class TestGetShardDir:
    def test_includes_dataset_and_vocab(self):

        path = get_shard_dir("fineweb-edu", "gpt-neo-125m")

        assert "fineweb-edu" in str(path)

        assert "vocab50257" in str(path)

    def test_base_dir_respected(self):

        path = get_shard_dir("wikitext-103", "gpt-neo-125m", base="/tmp/shards")

        assert str(path).startswith("/tmp/shards")

    def test_same_tokenizer_models_share_dir(self):

        path_125m = get_shard_dir("fineweb-edu", "gpt-neo-125m")

        path_1b = get_shard_dir("fineweb-edu", "gpt-neo-1.3b")

        assert path_125m == path_1b

    def test_returns_path_object(self):

        path = get_shard_dir("wikitext-2", "gpt-neo-125m")

        assert isinstance(path, Path)

    def test_path_structure(self):

        path = get_shard_dir("wikitext-2", "gpt-neo-125m", base="data/shards")

        parts = path.parts

        assert parts[-2] == "wikitext-2"

        assert parts[-1].startswith("vocab")


class TestDatasetConfig:
    def test_default_key_is_valid(self):

        cfg = DatasetConfig()

        info = cfg.get_dataset_info()

        assert "hf_path" in info

    def test_custom_dataset_bypasses_registry(self):

        cfg = DatasetConfig(
            custom_dataset_name="my/dataset", custom_dataset_config="v1"
        )

        info = cfg.get_dataset_info()

        assert info["hf_path"] == "my/dataset"

        assert info["hf_name"] == "v1"

    def test_unknown_key_raises(self):

        cfg = DatasetConfig(dataset_key="nonexistent")

        with pytest.raises(ValueError):
            cfg.get_dataset_info()

    def test_get_description_includes_seq_len(self):

        cfg = DatasetConfig(dataset_key="wikitext-2", max_seq_len=512)

        desc = cfg.get_description()

        assert "512" in desc
