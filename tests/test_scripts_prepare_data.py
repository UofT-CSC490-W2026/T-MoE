from unittest.mock import patch, MagicMock


def test_parse_args():
    from scripts.prepare_data import parse_args

    with patch("sys.argv", ["prepare_data.py", "--config", "test.yaml"]):
        args = parse_args()
        assert args.config == "test.yaml"
        assert args.dataset is None
        assert args.out_dir is None


def test_parse_args_with_overrides():
    from scripts.prepare_data import parse_args

    with patch(
        "sys.argv",
        [
            "prepare_data.py",
            "--config",
            "test.yaml",
            "--dataset",
            "wikitext-2",
            "--out-dir",
            "/tmp/out",
            "--num-proc",
            "4",
        ],
    ):
        args = parse_args()
        assert args.dataset == "wikitext-2"
        assert args.out_dir == "/tmp/out"
        assert args.num_proc == 4


def test_load_config(tmp_path):
    from scripts.prepare_data import load_config

    cfg_path = tmp_path / "test.yaml"

    cfg_path.write_text(
        "dataset:\n  dataset_key: wikitext-2\nmodel:\n  model_key: gpt-neo-125m\n"
    )

    cfg = load_config(str(cfg_path))

    assert cfg.dataset.dataset_key == "wikitext-2"


def test_load_config_with_dataset_override(tmp_path):
    from scripts.prepare_data import load_config

    cfg_path = tmp_path / "test.yaml"

    cfg_path.write_text(
        "dataset:\n  dataset_key: wikitext-2\nmodel:\n  model_key: gpt-neo-125m\n"
    )

    cfg = load_config(str(cfg_path), dataset_override="fineweb-edu")

    assert cfg.dataset.dataset_key == "fineweb-edu"


def test_get_tokenizer():
    from scripts.prepare_data import get_tokenizer

    mock_tok = MagicMock()

    mock_tok.pad_token = None

    mock_tok.eos_token = "<eos>"

    mock_tok.eos_token_id = 50256

    mock_tok.model_max_length = 512

    with patch(
        "src.configs.model.model_lookup",
        return_value={"hf_name": "EleutherAI/gpt-neo-125m"},
    ):
        with patch("transformers.AutoTokenizer") as MockTok:
            MockTok.from_pretrained.return_value = mock_tok
            tok, eos_id = get_tokenizer("gpt-neo-125m")
            assert eos_id == 50256


def test_tokenize_batch():
    from scripts.prepare_data import _tokenize_batch
    import scripts.prepare_data as pd_mod

    mock_tok = MagicMock()

    mock_tok.encode.return_value = [1, 2, 3]

    mock_tok.model_max_length = int(1e30)

    with patch.object(pd_mod, "_worker_tok", None):
        with patch.object(pd_mod, "_worker_tok_name", None):
            with patch("transformers.AutoTokenizer") as MockTok:
                MockTok.from_pretrained.return_value = mock_tok
                result = _tokenize_batch((["hello world", "  "], "gpt-neo-125m", 50256))
                assert len(result) == 1


def test_main_prepare_data(tmp_path):
    from scripts.prepare_data import main

    cfg_content = (
        "dataset:\n  dataset_key: wikitext-2\nmodel:\n  model_key: gpt-neo-125m\n"
    )

    cfg_path = tmp_path / "test.yaml"

    cfg_path.write_text(cfg_content)

    with patch(
        "sys.argv",
        ["prepare_data.py", "--config", str(cfg_path), "--out-dir", str(tmp_path)],
    ):
        with patch("scripts.prepare_data.tokenize_and_pack") as mock_pack:
            main()
            mock_pack.assert_called_once()


def test_main_prepare_data_with_dataset_override(tmp_path):
    from scripts.prepare_data import main

    cfg_content = (
        "dataset:\n  dataset_key: wikitext-2\nmodel:\n  model_key: gpt-neo-125m\n"
    )

    cfg_path = tmp_path / "test.yaml"

    cfg_path.write_text(cfg_content)

    with patch(
        "sys.argv",
        [
            "prepare_data.py",
            "--config",
            str(cfg_path),
            "--dataset",
            "fineweb-edu",
            "--out-dir",
            str(tmp_path),
        ],
    ):
        with patch("scripts.prepare_data.tokenize_and_pack") as mock_pack:
            main()
            mock_pack.assert_called_once()
