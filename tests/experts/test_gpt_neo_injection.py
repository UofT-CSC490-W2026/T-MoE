import pytest

import torch

from torch import nn

from unittest.mock import MagicMock, patch

from src.models.gpt_neo import GPTNeoBackbone


class MockGPTNeo(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.config = config

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.num_layers = config.num_layers

        self.hidden_size = config.hidden_size

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
        **kwargs,
    ):

        batch, seq = input_ids.shape

        hidden_states = [
            torch.randn(batch, seq, self.hidden_size)
            for _ in range(self.num_layers + 1)
        ]

        logits = torch.randn(batch, seq, self.config.vocab_size)

        loss = None

        if labels is not None:
            loss = torch.tensor(2.5)

        output = MagicMock()

        output.hidden_states = tuple(hidden_states) if output_hidden_states else None

        output.logits = logits

        output.loss = loss

        return output


@pytest.fixture
def mock_gpt_neo_components():

    with (
        patch("src.models.gpt_neo.AutoConfig") as mock_config_cls,
        patch("src.models.gpt_neo.AutoModelForCausalLM") as mock_model_cls,
    ):
        mock_config = MagicMock()

        mock_config.hidden_size = 64

        mock_config.vocab_size = 1000

        mock_config.num_layers = 4

        mock_config_cls.from_pretrained.return_value = mock_config

        mock_model = MockGPTNeo(mock_config)

        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_config, mock_model


@pytest.fixture
def gpt_neo_backbone(mock_gpt_neo_components):

    with patch.dict(
        GPTNeoBackbone.VARIANTS,
        {
            "125m": {
                "hf_name": "mock-gpt-neo",
                "hidden_dim": 64,
                "num_layers": 4,
                "num_heads": 4,
                "description": "Mock GPT-Neo",
            }
        },
    ):
        backbone = GPTNeoBackbone(
            variant="125m",
            freeze_backbone=True,
            moe_layer_indices=[-1],
            device="cpu",
        )

        return backbone


def test_initialization(mock_gpt_neo_components):

    mock_config, _ = mock_gpt_neo_components

    with patch.dict(
        GPTNeoBackbone.VARIANTS,
        {
            "125m": {
                "hf_name": "mock-gpt-neo",
                "hidden_dim": 64,
                "num_layers": 4,
                "num_heads": 4,
                "description": "Mock GPT-Neo",
            }
        },
    ):
        backbone = GPTNeoBackbone(
            variant="125m",
            freeze_backbone=True,
            moe_layer_indices=[-1],
            device="cpu",
        )

        assert backbone.hidden_dim == 64

        assert backbone.vocab_size == 1000

        assert backbone.num_layers == 4

        assert backbone.moe_layer_indices == [-1]

        assert backbone.backbone is not None


def test_forward_pass_basic(gpt_neo_backbone):

    input_ids = torch.randint(0, 1000, (2, 8))

    logits, loss, metrics = gpt_neo_backbone(input_ids=input_ids, return_metrics=False)

    assert logits.shape == (2, 8, 1000)

    assert loss is None

    assert metrics is None or metrics == {}


def test_forward_pass_with_labels(gpt_neo_backbone):

    input_ids = torch.randint(0, 1000, (2, 8))

    labels = torch.randint(0, 1000, (2, 8))

    logits, loss, metrics = gpt_neo_backbone(
        input_ids=input_ids, labels=labels, return_metrics=False
    )

    assert loss is not None

    assert isinstance(loss, torch.Tensor)

    assert not torch.isnan(loss)


def test_freeze_parameters(mock_gpt_neo_components):

    _, mock_model = mock_gpt_neo_components

    backbone = GPTNeoBackbone(variant="125m", freeze_backbone=True, device="cpu")

    for param in backbone.backbone.parameters():
        assert param.requires_grad is False
