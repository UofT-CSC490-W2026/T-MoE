from __future__ import annotations

import pytest

import torch.nn as nn

class FakeLoRAMoELayer(nn.Module):

    

    def __init__(self, hidden: int = 64):

        super().__init__()

        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):

        return self.linear(x)

class FakeAttention(nn.Module):

    

    def __init__(self, hidden: int = 64):

        super().__init__()

        self.q = nn.Linear(hidden, hidden)

    def forward(self, x):

        return self.q(x)

class FakeTransformerBlock(nn.Module):

    

    def __init__(self, hidden: int = 64):

        super().__init__()

        self.attn = FakeAttention(hidden)

        self.mlp = FakeLoRAMoELayer(hidden)                

    def forward(self, x):

        return self.mlp(self.attn(x))

class FakeModel(nn.Module):

    

    def __init__(self, n_layers: int = 2, hidden: int = 64):

        super().__init__()

        self.blocks = nn.ModuleList(

            [FakeTransformerBlock(hidden) for _ in range(n_layers)]

        )

        self.embed = nn.Embedding(100, hidden)

    def forward(self, x):

        h = self.embed(x)

        for block in self.blocks:

            h = block(h)

        return h

class TestModuleWrapPolicy:

    

    def _build_policy(self, cls):

        
        try:

            from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        except ImportError:

            pytest.skip("FSDP not available in this PyTorch build")

        return ModuleWrapPolicy({cls})

    def test_lora_moe_layer_is_matched(self):

        
        policy = self._build_policy(FakeLoRAMoELayer)

        moe = FakeLoRAMoELayer()

                                                                              

        assert policy(moe, recurse=False, nonwrapped_numel=0) is True

    def test_attention_not_matched(self):

        
        policy = self._build_policy(FakeLoRAMoELayer)

        attn = FakeAttention()

        assert policy(attn, recurse=False, nonwrapped_numel=0) is False

    def test_block_not_matched(self):

        
        policy = self._build_policy(FakeLoRAMoELayer)

        block = FakeTransformerBlock()

        assert policy(block, recurse=False, nonwrapped_numel=0) is False

    def test_linear_not_matched(self):

        
        policy = self._build_policy(FakeLoRAMoELayer)

        linear = nn.Linear(64, 64)

        assert policy(linear, recurse=False, nonwrapped_numel=0) is False

    def test_no_double_wrap_in_model_tree(self):

        
        policy = self._build_policy(FakeLoRAMoELayer)

        model = FakeModel(n_layers=3)

        wrapped_modules = []

        for name, module in model.named_modules():

            if policy(module, recurse=False, nonwrapped_numel=0):

                wrapped_modules.append((name, module))

                                                                                 

        wrapped_set = {id(m) for _, m in wrapped_modules}

        for wrap_name, wrap_module in wrapped_modules:

            for child_name, child in wrap_module.named_children():

                assert id(child) not in wrapped_set, (

                    f"Double-wrap detected: {wrap_name}.{child_name} would be wrapped "

                    f"as a child of already-wrapped module {wrap_name}"

                )

    def test_string_matching_would_cause_double_wrap(self):

        

                                       

        def old_buggy_policy(module, recurse, nonwrapped_numel):

            name = module.__class__.__name__

            return "Block" in name or "Layer" in name

        model = FakeModel(n_layers=2)

        matched = [

            (n, m) for n, m in model.named_modules() if old_buggy_policy(m, False, 0)

        ]

                                                 

                                                                                    

        has_block = any("FakeTransformerBlock" in type(m).__name__ for _, m in matched)

        has_moe = any("FakeLoRAMoELayer" in type(m).__name__ for _, m in matched)

        assert has_block and has_moe, (

            "Expected old policy to match both Block and MoE (double-wrap scenario)"

        )

                                                          

        policy = self._build_policy(FakeLoRAMoELayer)

        new_matched = [(n, m) for n, m in model.named_modules() if policy(m, False, 0)]

        new_has_block = any(

            "FakeTransformerBlock" in type(m).__name__ for _, m in new_matched

        )

        assert not new_has_block, (

            "New policy should NOT match FakeTransformerBlock — only LoRAMoELayer"

        )

class TestDistributedUtils:

    def test_is_main_process_returns_true_when_not_distributed(self):

        
        from src.training.fsdp_utils import is_main_process

                                                

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():

            pytest.skip("Process group already initialized — skip in CI")

        assert is_main_process() is True

    def test_init_distributed_returns_false_without_env(self, monkeypatch):

        
        monkeypatch.delenv("RANK", raising=False)

        from src.training.fsdp_utils import init_distributed

        result = init_distributed()

        assert result == (False, 0, 0, 1)

    def test_get_model_for_attr_access_passthrough(self):

        
        from src.training.fsdp_utils import get_model_for_attr_access

        model = nn.Linear(10, 10)

        assert get_model_for_attr_access(model) is model
