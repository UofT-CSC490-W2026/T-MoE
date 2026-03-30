"""
SPAR training entrypoint.

Single-GPU:
    python train.py --config experiments/qwen2_1.5b_stress_v3.yaml

Multi-GPU (FSDP via torchrun):
    torchrun --standalone --nproc_per_node=4 train.py \\
        --config experiments/qwen2_1.5b_stress_v3.yaml
"""

from scripts.train import main

if __name__ == "__main__":
    main()
