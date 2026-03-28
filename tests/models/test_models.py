import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


def test_base_model_abstract():
    from src.models.base import BaseModelBackbone

    with pytest.raises(TypeError):
        BaseModelBackbone("test", 768)


def test_base_model_inject_moe_layers():
    from src.models.base import BaseModelBackbone

    class ConcreteModel(BaseModelBackbone):
        def load_pretrained(self):
            pass

        def forward(self, *a, **kw):
            pass

    model = ConcreteModel("test", 768)
    mock_layer = MagicMock()
    model.inject_moe_layers({0: mock_layer})
    assert "0" in model.moe_layers


def test_base_model_freeze_unfreeze():
    from src.models.base import BaseModelBackbone

    class ConcreteModel(BaseModelBackbone):
        def load_pretrained(self):
            pass

        def forward(self, *a, **kw):
            pass

    model = ConcreteModel("test", 768)
    model.backbone = nn.Linear(10, 10)
    model.freeze_parameters()
    assert not model.backbone.weight.requires_grad
    model.unfreeze_parameters()
    assert model.backbone.weight.requires_grad


def test_base_model_param_counts():
    from src.models.base import BaseModelBackbone

    class ConcreteModel(BaseModelBackbone):
        def load_pretrained(self):
            pass

        def forward(self, *a, **kw):
            pass

    model = ConcreteModel("test", 768)
    model.backbone = nn.Linear(10, 10)
    total = model.get_total_params()
    trainable = model.get_trainable_params()
    assert total > 0
    assert trainable > 0


def test_base_model_get_mlp_at_not_implemented():
    from src.models.base import BaseModelBackbone

    class ConcreteModel(BaseModelBackbone):
        def load_pretrained(self):
            pass

        def forward(self, *a, **kw):
            pass

    model = ConcreteModel("test", 768)
    with pytest.raises(NotImplementedError):
        model.get_mlp_at(0)


def _make_mock_gptneo():
    mock_backbone = MagicMock()
    mock_backbone.config.vocab_size = 50257
    blocks = [MagicMock() for _ in range(12)]
    for block in blocks:
        block.mlp = nn.Linear(768, 768)
    mock_backbone.transformer.h = blocks
    mock_backbone.parameters.return_value = iter([torch.randn(10, 10)])
    mock_backbone.to.return_value = mock_backbone
    return mock_backbone


def test_gptneo_invalid_variant():
    from src.models.gpt_neo import GPTNeoBackbone

    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = _make_mock_gptneo()
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            with pytest.raises(ValueError, match="Invalid variant"):
                GPTNeoBackbone(variant="999b")


def test_gptneo_get_variant_info():
    from src.models.gpt_neo import GPTNeoBackbone

    info = GPTNeoBackbone.get_variant_info("125m")
    assert info["hidden_dim"] == 768


def test_gptneo_get_variant_info_invalid():
    from src.models.gpt_neo import GPTNeoBackbone

    with pytest.raises(ValueError):
        GPTNeoBackbone.get_variant_info("invalid")


def test_gptneo_list_variants():
    from src.models.gpt_neo import GPTNeoBackbone

    variants = GPTNeoBackbone.list_variants()
    assert "125m" in variants


def test_gptneo_backbone_init():
    from src.models.gpt_neo import GPTNeoBackbone

    mock_backbone = _make_mock_gptneo()
    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            model = GPTNeoBackbone(variant="125m", freeze_backbone=True, device="cpu")
    assert model.vocab_size == 50257
    assert model.num_layers == 12


def test_gptneo_get_mlp_at():
    from src.models.gpt_neo import GPTNeoBackbone

    mock_backbone = _make_mock_gptneo()
    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            model = GPTNeoBackbone(variant="125m", device="cpu")
    mlp = model.get_mlp_at(0)
    assert mlp is not None


def test_gptneo_inject_moe_layers():
    from src.models.gpt_neo import GPTNeoBackbone

    mock_backbone = _make_mock_gptneo()
    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            model = GPTNeoBackbone(variant="125m", device="cpu")
    mock_moe = MagicMock()
    model.inject_moe_layers({0: mock_moe})
    assert "0" in model.moe_layers


def test_gptneo_inject_moe_layers_invalid_idx():
    from src.models.gpt_neo import GPTNeoBackbone

    mock_backbone = _make_mock_gptneo()
    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            model = GPTNeoBackbone(variant="125m", device="cpu")
    with pytest.raises(ValueError, match="Invalid layer index"):
        model.inject_moe_layers({99: MagicMock()})


def test_gptneo_inject_moe_layers_empty():
    from src.models.gpt_neo import GPTNeoBackbone

    mock_backbone = _make_mock_gptneo()
    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            model = GPTNeoBackbone(variant="125m", device="cpu")
    model.inject_moe_layers({})


def test_gptneo_forward():
    from src.models.gpt_neo import GPTNeoBackbone

    mock_backbone = _make_mock_gptneo()
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.randn(2, 10, 50257)
    mock_outputs.loss = None
    mock_backbone.return_value = mock_outputs
    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            model = GPTNeoBackbone(variant="125m", device="cpu")
    model.backbone.return_value = mock_outputs
    input_ids = torch.randint(0, 50257, (2, 10))
    logits, loss, metrics = model(input_ids)
    assert logits.shape == (2, 10, 50257)
    assert loss is None


def test_gptneo_forward_with_labels():
    from src.models.gpt_neo import GPTNeoBackbone

    mock_backbone = _make_mock_gptneo()
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.randn(2, 10, 50257)
    mock_outputs.loss = torch.tensor(1.5)
    mock_backbone.return_value = mock_outputs
    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            model = GPTNeoBackbone(variant="125m", device="cpu")
    model.backbone.return_value = mock_outputs
    input_ids = torch.randint(0, 50257, (2, 10))
    labels = torch.randint(0, 50257, (2, 10))
    logits, loss, metrics = model(input_ids, labels=labels)
    assert loss is not None


def test_gptneo_forward_with_moe_and_metrics():
    from src.models.gpt_neo import GPTNeoBackbone

    mock_backbone = _make_mock_gptneo()
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.randn(2, 10, 50257)
    mock_outputs.loss = torch.tensor(1.5)
    with patch("src.models.gpt_neo.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch("src.models.gpt_neo.AutoConfig") as MockCfg:
            MockCfg.from_pretrained.return_value = MagicMock(vocab_size=50257)
            model = GPTNeoBackbone(variant="125m", device="cpu")
    model.backbone.return_value = mock_outputs
    mock_moe = MagicMock()
    mock_moe.get_cached_metrics.return_value = {"entropy": 1.5}
    mock_moe.router.compute_aux_loss.return_value = torch.tensor(0.0)
    model.moe_layers = {"0": mock_moe}
    input_ids = torch.randint(0, 50257, (2, 10))
    labels = torch.randint(0, 50257, (2, 10))
    logits, loss, metrics = model(input_ids, labels=labels, return_metrics=True)
    assert metrics is not None


def _make_mock_qwen2():
    mock_backbone = MagicMock()
    mock_backbone.config.vocab_size = 151936
    layers = [MagicMock() for _ in range(28)]
    for block in layers:
        block.mlp = nn.Linear(1536, 1536)
    mock_backbone.model.layers = layers
    mock_backbone.to.return_value = mock_backbone
    return mock_backbone


def test_qwen2_invalid_variant():
    from src.models.qwen2 import Qwen2Backbone

    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = _make_mock_qwen2()
        with pytest.raises(ValueError, match="Invalid variant"):
            Qwen2Backbone(variant="invalid")


def test_qwen2_backbone_init():
    from src.models.qwen2 import Qwen2Backbone

    mock_backbone = _make_mock_qwen2()
    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        model = Qwen2Backbone(variant="1.5b", freeze_backbone=True, device="cpu")
    assert model.num_layers == 28


def test_qwen2_get_mlp_at():
    from src.models.qwen2 import Qwen2Backbone

    mock_backbone = _make_mock_qwen2()
    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        model = Qwen2Backbone(variant="1.5b", device="cpu")
    mlp = model.get_mlp_at(0)
    assert mlp is not None


def test_qwen2_inject_moe_layers():
    from src.models.qwen2 import Qwen2Backbone

    mock_backbone = _make_mock_qwen2()
    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        model = Qwen2Backbone(variant="1.5b", device="cpu")
    mock_moe = MagicMock()
    model.inject_moe_layers({0: mock_moe})
    assert "0" in model.moe_layers


def test_qwen2_inject_moe_layers_invalid():
    from src.models.qwen2 import Qwen2Backbone

    mock_backbone = _make_mock_qwen2()
    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        model = Qwen2Backbone(variant="1.5b", device="cpu")
    with pytest.raises(ValueError, match="Invalid layer index"):
        model.inject_moe_layers({99: MagicMock()})


def test_qwen2_inject_moe_layers_empty():
    from src.models.qwen2 import Qwen2Backbone

    mock_backbone = _make_mock_qwen2()
    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        model = Qwen2Backbone(variant="1.5b", device="cpu")
    model.inject_moe_layers({})


def test_qwen2_forward():
    from src.models.qwen2 import Qwen2Backbone

    mock_backbone = _make_mock_qwen2()
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.randn(2, 10, 151936)
    mock_outputs.loss = None
    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        model = Qwen2Backbone(variant="1.5b", device="cpu")
    model.backbone.return_value = mock_outputs
    input_ids = torch.randint(0, 151936, (2, 10))
    logits, loss, metrics = model(input_ids)
    assert logits.shape[0] == 2


def test_qwen2_forward_with_labels_and_moe():
    from src.models.qwen2 import Qwen2Backbone

    mock_backbone = _make_mock_qwen2()
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.randn(2, 10, 151936)
    mock_outputs.loss = torch.tensor(2.0)
    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        model = Qwen2Backbone(variant="1.5b", device="cpu")
    model.backbone.return_value = mock_outputs
    mock_moe = MagicMock()
    mock_moe.get_cached_metrics.return_value = {"entropy": 1.5}
    mock_moe.router.compute_aux_loss.return_value = torch.tensor(0.0)
    model.moe_layers = {"0": mock_moe}
    input_ids = torch.randint(0, 151936, (2, 10))
    labels = torch.randint(0, 151936, (2, 10))
    logits, loss, metrics = model(input_ids, labels=labels, return_metrics=True)
    assert loss is not None
    assert metrics is not None


def test_qwen2_load_pretrained_flash_attn():
    from src.models.qwen2 import Qwen2Backbone

    mock_backbone = _make_mock_qwen2()
    with patch("src.models.qwen2.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = mock_backbone
        with patch.dict("sys.modules", {"flash_attn": MagicMock()}):
            with patch("src.models.qwen2.torch.cuda.is_available", return_value=False):
                model = Qwen2Backbone(variant="1.5b", device="cpu")
    assert model is not None
