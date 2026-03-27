from unittest.mock import patch


def test_train_entrypoint_imports():
    pass


def test_train_entrypoint_main_not_called_on_import():
    with patch("scripts.train.main") as mock_main:
        import importlib
        import train

        importlib.reload(train)
        mock_main.assert_not_called()
