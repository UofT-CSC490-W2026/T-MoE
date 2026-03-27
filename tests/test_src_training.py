import pytest

import torch

import torch.nn as nn

from unittest.mock import patch, MagicMock

                                                                                 

def test_detect_dtype_env_bf16(monkeypatch):

    monkeypatch.setenv("TMOE_DTYPE", "bfloat16")

    import importlib

    import src.training.precision as prec

    importlib.reload(prec)

    assert prec.COMPUTE_DTYPE == torch.bfloat16

    monkeypatch.delenv("TMOE_DTYPE")

    importlib.reload(prec)

def test_detect_dtype_env_fp32(monkeypatch):

    monkeypatch.setenv("TMOE_DTYPE", "fp32")

    import importlib

    import src.training.precision as prec

    importlib.reload(prec)

    assert prec.COMPUTE_DTYPE == torch.float32

    monkeypatch.delenv("TMOE_DTYPE")

    importlib.reload(prec)

def test_detect_dtype_env_fp16(monkeypatch):

    monkeypatch.setenv("TMOE_DTYPE", "fp16")

    import importlib

    import src.training.precision as prec

    importlib.reload(prec)

    assert prec.COMPUTE_DTYPE == torch.float16

    monkeypatch.delenv("TMOE_DTYPE")

    importlib.reload(prec)

def test_detect_dtype_env_bf16_alias(monkeypatch):

    monkeypatch.setenv("TMOE_DTYPE", "bf16")

    import importlib

    import src.training.precision as prec

    importlib.reload(prec)

    assert prec.COMPUTE_DTYPE == torch.bfloat16

    monkeypatch.delenv("TMOE_DTYPE")

    importlib.reload(prec)

def test_detect_dtype_invalid_env(monkeypatch):

    monkeypatch.setenv("TMOE_DTYPE", "invalid_dtype")

    import importlib

    import src.training.precision as prec

    with pytest.raises(ValueError, match="Unknown TMOE_DTYPE"):

        prec._detect_dtype()

    monkeypatch.delenv("TMOE_DTYPE")

    importlib.reload(prec)

def test_is_mixed_precision():

    from src.training.precision import is_mixed_precision, COMPUTE_DTYPE

    result = is_mixed_precision()

    assert isinstance(result, bool)

    if COMPUTE_DTYPE in (torch.bfloat16, torch.float16):

        assert result is True

    else:

        assert result is False

def test_needs_grad_scaler():

    from src.training.precision import needs_grad_scaler, COMPUTE_DTYPE

    result = needs_grad_scaler()

    assert isinstance(result, bool)

    if COMPUTE_DTYPE == torch.float16:

        assert result is True

    else:

        assert result is False

                                                                                 

def test_cleanup_distributed_not_initialized():

    from src.training.fsdp_utils import cleanup_distributed

                                                

    cleanup_distributed()

def test_get_model_for_attr_access_ddp():

    from src.training.fsdp_utils import get_model_for_attr_access

    from torch.nn.parallel import DistributedDataParallel as DDP

    model = nn.Linear(10, 10)

                                    

    mock_ddp = MagicMock(spec=DDP)

    mock_ddp.module = model

    result = get_model_for_attr_access(mock_ddp)

    assert result is model

def test_wrap_model_for_distributed_ddp_strategy():

    
    from src.training.fsdp_utils import wrap_model_for_distributed

    model = nn.Linear(10, 10)

    cfg = MagicMock()

    dist_cfg = MagicMock()

    dist_cfg.strategy = "ddp"

    cfg.distributed = dist_cfg

    with patch("src.training.fsdp_utils.wrap_model_with_ddp") as mock_ddp:

        mock_ddp.return_value = model

        wrap_model_for_distributed(model, cfg, 0, torch.device("cpu"))

        mock_ddp.assert_called_once()

def test_wrap_model_for_distributed_fsdp_strategy():

    
    from src.training.fsdp_utils import wrap_model_for_distributed

    model = nn.Linear(10, 10)

    cfg = MagicMock()

    dist_cfg = MagicMock()

    dist_cfg.strategy = "fsdp"

    cfg.distributed = dist_cfg

    with patch("src.training.fsdp_utils.wrap_model_with_fsdp") as mock_fsdp:

        mock_fsdp.return_value = model

        wrap_model_for_distributed(model, cfg, 0, torch.device("cpu"))

        mock_fsdp.assert_called_once()

def test_wrap_model_for_distributed_no_strategy():

    
    from src.training.fsdp_utils import wrap_model_for_distributed

    model = nn.Linear(10, 10)

    cfg = MagicMock()

    cfg.distributed = {}

    with patch("src.training.fsdp_utils.wrap_model_with_ddp") as mock_ddp:

        mock_ddp.return_value = model

        wrap_model_for_distributed(model, cfg, 0, torch.device("cpu"))

        mock_ddp.assert_called_once()

def test_init_distributed_no_cuda(monkeypatch):

    monkeypatch.setenv("RANK", "0")

    monkeypatch.setenv("LOCAL_RANK", "0")

    monkeypatch.setenv("WORLD_SIZE", "1")

    from src.training.fsdp_utils import init_distributed

    with patch("src.training.fsdp_utils.torch.cuda.is_available", return_value=False):

        with pytest.raises(RuntimeError, match="CUDA"):

            init_distributed()

    monkeypatch.delenv("RANK")

    monkeypatch.delenv("LOCAL_RANK")

    monkeypatch.delenv("WORLD_SIZE")

def test_init_distributed_local_rank_too_high(monkeypatch):

    monkeypatch.setenv("RANK", "0")

    monkeypatch.setenv("LOCAL_RANK", "99")

    monkeypatch.setenv("WORLD_SIZE", "1")

    from src.training.fsdp_utils import init_distributed

    with patch("src.training.fsdp_utils.torch.cuda.is_available", return_value=True):

        with patch("src.training.fsdp_utils.torch.cuda.device_count", return_value=1):

            with pytest.raises(RuntimeError, match="LOCAL_RANK"):

                init_distributed()

    monkeypatch.delenv("RANK")

    monkeypatch.delenv("LOCAL_RANK")

    monkeypatch.delenv("WORLD_SIZE")

                                                                                 

def test_serialize_metrics():

    from src.training.checkpoint import _serialize_metrics

    result = _serialize_metrics({"loss": 0.5, "step": 100, "name": "test"})

    assert result["loss"] == 0.5

    assert result["step"] == 100.0

def test_remap_legacy_moe_key_router():

    from src.training.checkpoint import _remap_legacy_moe_key

    key = "moe_layers.11.router.gate.weight"

    result = _remap_legacy_moe_key(key)

    assert "backbone.transformer.h.11.mlp.router" in result

def test_remap_legacy_moe_key_experts_fc1():

    from src.training.checkpoint import _remap_legacy_moe_key

    key = "moe_layers.11.experts.0.fc1.lora_A.weight"

    result = _remap_legacy_moe_key(key)

    assert "c_fc" in result

def test_remap_legacy_moe_key_experts_fc2():

    from src.training.checkpoint import _remap_legacy_moe_key

    key = "moe_layers.11.experts.0.fc2.lora_B.weight"

    result = _remap_legacy_moe_key(key)

    assert "c_proj" in result

def test_remap_legacy_moe_key_base_weight_dropped():

    from src.training.checkpoint import _remap_legacy_moe_key

    key = "moe_layers.11.experts.0.fc1.base_weight"

    result = _remap_legacy_moe_key(key)

    assert result is None

def test_remap_legacy_moe_key_mlp_experts():

    from src.training.checkpoint import _remap_legacy_moe_key

    key = "backbone.transformer.h.11.mlp.experts.0.fc1.lora_A.weight"

    result = _remap_legacy_moe_key(key)

    assert result is not None

def test_remap_legacy_moe_key_passthrough():

    from src.training.checkpoint import _remap_legacy_moe_key

    key = "backbone.transformer.h.0.attn.weight"

    result = _remap_legacy_moe_key(key)

    assert result == key

def test_remap_legacy_moe_key_short():

    from src.training.checkpoint import _remap_legacy_moe_key

    key = "moe_layers.11"

    result = _remap_legacy_moe_key(key)

    assert result == key

def test_remap_legacy_moe_state_dict():

    from src.training.checkpoint import _remap_legacy_moe_state_dict

    state = {

        "moe_layers.11.router.gate.weight": torch.randn(8, 768),

        "moe_layers.11.experts.0.fc1.base_weight": torch.randn(3072, 768),

        "backbone.other.weight": torch.randn(10, 10),

    }

    remapped, changed = _remap_legacy_moe_state_dict(state)

    assert changed is True

    assert "backbone.other.weight" in remapped

def test_log_state_dict_result_not_main():

    from src.training.checkpoint import _log_state_dict_result

    result = MagicMock()

    result.missing_keys = ["key1"]

    result.unexpected_keys = ["key2"]

    with patch("src.training.checkpoint.is_main_process", return_value=False):

        _log_state_dict_result(result, "test")                    

def test_log_state_dict_result_main(capsys):

    from src.training.checkpoint import _log_state_dict_result

    result = MagicMock()

    result.missing_keys = ["key1"]

    result.unexpected_keys = ["key2"]

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        _log_state_dict_result(result, "test")

def test_checkpoint_manager_save_load(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(tmp_path), keep_last_n=3, save_best=True)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch("torch.distributed.is_initialized", return_value=False):

            path = manager.save_checkpoint(

                model, optimizer, step=10, metrics={"loss": 0.5}, is_best=True

            )

    assert path.exists()

                  

    model2 = nn.Linear(10, 10)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        info = manager.load_checkpoint(model2, checkpoint_path=path)

    assert info["step"] == 10

    assert info["metrics"]["loss"] == 0.5

def test_checkpoint_manager_save_non_main(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(tmp_path))

    with patch("src.training.checkpoint.is_main_process", return_value=False):

        with patch("torch.distributed.is_initialized", return_value=False):

            path = manager.save_checkpoint(model, optimizer, step=5)

    assert str(path) == "/dev/null"

def test_checkpoint_manager_load_best(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(tmp_path), save_best=True)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch("torch.distributed.is_initialized", return_value=False):

            manager.save_checkpoint(

                model, optimizer, step=10, metrics={"loss": 0.3}, is_best=True

            )

    model2 = nn.Linear(10, 10)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        info = manager.load_checkpoint(model2, load_best=True)

    assert info["step"] == 10

def test_checkpoint_manager_load_no_checkpoint(tmp_path):

    from src.training.checkpoint import CheckpointManager

    manager = CheckpointManager(str(tmp_path))

    model = nn.Linear(10, 10)

    with pytest.raises(FileNotFoundError):

        manager.load_checkpoint(model)

def test_checkpoint_manager_cleanup(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(tmp_path), keep_last_n=2)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch("torch.distributed.is_initialized", return_value=False):

            for step in [10, 20, 30]:

                manager.save_checkpoint(

                    model, optimizer, step=step, metrics={"loss": 0.5}

                )

                                      

    remaining = list(tmp_path.glob("checkpoint_step_*.pt"))

    assert len(remaining) == 2

def test_checkpoint_manager_list(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(tmp_path))

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch("torch.distributed.is_initialized", return_value=False):

            manager.save_checkpoint(model, optimizer, step=5)

    listing = manager.list_checkpoints()

    assert len(listing) == 1

    assert listing[0]["step"] == 5

def test_checkpoint_manager_trainable_only(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    model.weight.requires_grad = True

    model.bias.requires_grad = False

    optimizer = torch.optim.Adam([model.weight], lr=1e-3)

    manager = CheckpointManager(str(tmp_path), trainable_only=True)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch(

            "src.training.checkpoint.get_model_for_attr_access", return_value=model

        ):

            with patch("torch.distributed.is_initialized", return_value=False):

                path = manager.save_checkpoint(model, optimizer, step=1)

    assert path.exists()

def test_checkpoint_manager_with_scheduler(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    manager = CheckpointManager(str(tmp_path))

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch("torch.distributed.is_initialized", return_value=False):

            path = manager.save_checkpoint(

                model, optimizer, scheduler=scheduler, step=1

            )

    model2 = nn.Linear(10, 10)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        info = manager.load_checkpoint(

            model2, optimizer2, scheduler2, checkpoint_path=path

        )

    assert info["step"] == 1

def test_checkpoint_manager_get_latest_from_dir(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(tmp_path))

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch("torch.distributed.is_initialized", return_value=False):

            manager.save_checkpoint(model, optimizer, step=100)

            manager.save_checkpoint(model, optimizer, step=200)

                                                                   

    manager2 = CheckpointManager(str(tmp_path))

    latest = manager2._get_latest_checkpoint()

    assert latest is not None

    assert "200" in str(latest)

def test_checkpoint_manager_keep_last_n_zero(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(tmp_path), keep_last_n=0)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch("torch.distributed.is_initialized", return_value=False):

            for step in [10, 20, 30]:

                manager.save_checkpoint(model, optimizer, step=step)

                                                      

    remaining = list(tmp_path.glob("checkpoint_step_*.pt"))

    assert len(remaining) == 3

def test_get_state_dict_plain():

    from src.training.checkpoint import _get_state_dict

    model = nn.Linear(10, 10)

    sd = _get_state_dict(model)

    assert "weight" in sd

def test_get_state_dict_ddp():

    from src.training.checkpoint import _get_state_dict

    from torch.nn.parallel import DistributedDataParallel as DDP

    model = nn.Linear(10, 10)

    mock_ddp = MagicMock(spec=DDP)

    mock_ddp.module = model

    sd = _get_state_dict(mock_ddp)

    assert "weight" in sd

                                                                                

def test_detect_dtype_cuda_sm8(monkeypatch):

    
    monkeypatch.delenv("TMOE_DTYPE", raising=False)

    import importlib

    import src.training.precision as prec

    with patch("torch.cuda.is_available", return_value=True):

        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):

            dtype = prec._detect_dtype()

    assert dtype == torch.bfloat16

    importlib.reload(prec)

def test_detect_dtype_cuda_sm7(monkeypatch):

    
    monkeypatch.delenv("TMOE_DTYPE", raising=False)

    import importlib

    import src.training.precision as prec

    with patch("torch.cuda.is_available", return_value=True):

        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):

            dtype = prec._detect_dtype()

    assert dtype == torch.float16

    importlib.reload(prec)

                                                                                

def test_remap_legacy_moe_key_unknown_block():

    from src.training.checkpoint import _remap_legacy_moe_key

                                                        

    key = "moe_layers.11.experts.0.unknown_block.weight"

    result = _remap_legacy_moe_key(key)

    assert result is not None

def test_remap_legacy_moe_key_mlp_experts_base_weight():

    from src.training.checkpoint import _remap_legacy_moe_key

    key = "backbone.transformer.h.11.mlp.experts.0.fc1.base_weight"

    result = _remap_legacy_moe_key(key)

    assert result is None

def test_checkpoint_load_with_legacy_keys(tmp_path):

    from src.training.checkpoint import CheckpointManager

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(tmp_path))

                       

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        with patch("torch.distributed.is_initialized", return_value=False):

            path = manager.save_checkpoint(model, optimizer, step=1)

                                                            

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    ckpt["model_state_dict"]["moe_layers.0.router.gate.weight"] = torch.randn(4, 10)

    torch.save(ckpt, path)

    model2 = nn.Linear(10, 10)

    with patch("src.training.checkpoint.is_main_process", return_value=True):

        info = manager.load_checkpoint(model2, checkpoint_path=path)

    assert info["step"] == 1

                                                                                

def test_remap_legacy_moe_key_short_suffix():

    
    from src.training.checkpoint import _remap_legacy_moe_key

                                                                               

    key = "moe_layers.11.experts.0.weight"

    result = _remap_legacy_moe_key(key)

    assert result is not None

    assert "0" in result

def test_checkpoint_manager_get_latest_from_memory():

    
    from src.training.checkpoint import CheckpointManager

    import torch.nn as nn

    model = nn.Linear(10, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:

        manager = CheckpointManager(tmp)

        with patch("src.training.checkpoint.is_main_process", return_value=True):

            with patch("torch.distributed.is_initialized", return_value=False):

                manager.save_checkpoint(model, optimizer, step=42)

        latest = manager._get_latest_checkpoint()

        assert latest is not None

        assert "42" in str(latest)

                                                                                

def test_switch_router_forces_top_k_1():

    
    from src.routers.standard import SwitchRouter

    from src.configs.router import SwitchRouterConfig

    cfg = SwitchRouterConfig(hidden_dim=64, num_experts=4, top_k=2)

    router = SwitchRouter(cfg)

    assert router.top_k == 1
