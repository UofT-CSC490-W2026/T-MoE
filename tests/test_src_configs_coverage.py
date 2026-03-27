import pytest


def test_get_dataset_info_valid():
    from src.configs.dataset import get_dataset_info

    info = get_dataset_info("wikitext-2")

    assert info["hf_path"] == "wikitext"


def test_get_dataset_info_invalid():
    from src.configs.dataset import get_dataset_info

    with pytest.raises(ValueError, match="Unknown dataset key"):
        get_dataset_info("nonexistent-dataset")


def test_get_shard_dir():
    from src.configs.dataset import get_shard_dir

    path = get_shard_dir("wikitext-2", "gpt-neo-125m", base="/tmp/shards")

    assert "wikitext-2" in str(path)

    assert "vocab50257" in str(path)


def test_dataset_config_get_dataset_info_custom():
    from src.configs.dataset import DatasetConfig

    cfg = DatasetConfig(
        custom_dataset_name="my/dataset", custom_dataset_config="config1"
    )

    info = cfg.get_dataset_info()

    assert info["hf_path"] == "my/dataset"

    assert info["hf_name"] == "config1"


def test_dataset_config_get_dataset_info_registry():
    from src.configs.dataset import DatasetConfig

    cfg = DatasetConfig(dataset_key="wikitext-2")

    info = cfg.get_dataset_info()

    assert "hf_path" in info


def test_dataset_config_get_description_custom():
    from src.configs.dataset import DatasetConfig

    cfg = DatasetConfig(custom_dataset_name="my/dataset", max_seq_len=512)

    desc = cfg.get_description()

    assert "my/dataset" in desc

    assert "512" in desc


def test_dataset_config_get_description_registry():
    from src.configs.dataset import DatasetConfig

    cfg = DatasetConfig(dataset_key="wikitext-2", max_seq_len=1024)

    desc = cfg.get_description()

    assert "wikitext" in desc


def test_dataset_config_get_description_with_hf_name():
    from src.configs.dataset import DatasetConfig

    cfg = DatasetConfig(dataset_key="fineweb-edu", max_seq_len=512)

    desc = cfg.get_description()

    assert "fineweb" in desc.lower()


def test_model_lookup_valid():
    from src.configs.model import model_lookup

    info = model_lookup("gpt-neo-125m")

    assert info["hidden_dim"] == 768

    assert info["model_type"] == "gpt_neo"


def test_model_lookup_invalid():
    from src.configs.model import model_lookup

    with pytest.raises(ValueError, match="Unknown model_key"):
        model_lookup("nonexistent-model-xyz")


def test_model_lookup_by_variant_key():
    from src.configs.model import model_lookup

    info = model_lookup("125m")

    assert info["hidden_dim"] == 768


def test_model_config_get_model_info():
    from src.configs.model import ModelConfig

    cfg = ModelConfig(model_key="gpt-neo-125m")

    info = cfg.get_model_info()

    assert info["hidden_dim"] == 768


def test_model_config_get_description():
    from src.configs.model import ModelConfig

    cfg = ModelConfig(model_key="gpt-neo-125m")

    desc = cfg.get_description()

    assert "768" in desc


def test_model_config_defaults():
    from src.configs.model import ModelConfig

    cfg = ModelConfig()

    assert cfg.model_key == "gpt-neo-125m"

    assert cfg.freeze_backbone is True

    assert cfg.device == "cuda"
