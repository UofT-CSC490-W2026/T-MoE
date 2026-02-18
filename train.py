import argparse
from src.training.main import train_main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="T-MoE Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Name of the experiment configuration file in the experiments/ directory (without .yaml extension)",
    )

    # Allow passing extra arguments to override config
    args, overrides = parser.parse_known_args()

    # Run the main training logic
    train_main(args, overrides)
