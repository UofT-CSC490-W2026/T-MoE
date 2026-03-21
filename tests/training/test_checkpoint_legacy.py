from src.training.checkpoint import _remap_legacy_moe_state_dict


def test_remap_legacy_moe_state_dict_maps_router_and_lora_keys():
    legacy_state_dict = {
        "moe_layers.1.router.gate.weight": "router_weight",
        "moe_layers.1.experts.0.fc1.lora_A.weight": "fc1_a",
        "moe_layers.1.experts.0.fc1.lora_B.weight": "fc1_b",
        "moe_layers.1.experts.0.fc2.lora_A.weight": "fc2_a",
        "moe_layers.1.experts.0.fc2.lora_B.weight": "fc2_b",
        "moe_layers.1.experts.0.fc1.base_weight": "drop_me",
        "backbone.transformer.h.3.mlp.experts.2.fc1.lora_A.weight": "already_backbone_fc1_a",
        "backbone.transformer.h.3.mlp.experts.2.fc2.lora_B.weight": "already_backbone_fc2_b",
        "backbone.transformer.wte.weight": "token_embed",
    }

    remapped, changed = _remap_legacy_moe_state_dict(legacy_state_dict)

    assert changed is True
    assert (
        remapped["backbone.transformer.h.1.mlp.router.gate.weight"] == "router_weight"
    )
    assert (
        remapped[
            "backbone.transformer.h.1.mlp.expert_pool.experts.0.c_fc.lora_A.weight"
        ]
        == "fc1_a"
    )
    assert (
        remapped[
            "backbone.transformer.h.1.mlp.expert_pool.experts.0.c_proj.lora_B.weight"
        ]
        == "fc2_b"
    )
    assert (
        remapped[
            "backbone.transformer.h.3.mlp.expert_pool.experts.2.c_fc.lora_A.weight"
        ]
        == "already_backbone_fc1_a"
    )
    assert (
        remapped[
            "backbone.transformer.h.3.mlp.expert_pool.experts.2.c_proj.lora_B.weight"
        ]
        == "already_backbone_fc2_b"
    )
    assert "moe_layers.1.experts.0.fc1.base_weight" not in remapped
    assert remapped["backbone.transformer.wte.weight"] == "token_embed"
