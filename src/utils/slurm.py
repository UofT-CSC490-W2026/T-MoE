import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


def generate_sbatch_script(config: DictConfig, config_name: str) -> Optional[str]:
    """
    Generate SBATCH script if running locally with SLURM enabled.
    """
    if config.execution_env != "local":
        return None

    if not config.compute.local.slurm.enabled:
        return None

    if not config.compute.local.slurm.auto_generate_script:
        return None

    # Create scripts directory
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)

    # Generate script path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = scripts_dir / f"{config.experiment_name}_{timestamp}.sh"

    # Get SLURM config
    slurm = config.compute.local.slurm
    output_root = config.compute.local.output_root
    log_dir = Path(output_root) / "logs" / config.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate SBATCH script
    script_content = f"""#!/bin/bash
#
# T-MoE Experiment: {config.experiment_name}
# Auto-generated: {timestamp}
#
#SBATCH --job-name={config.experiment_name}
#SBATCH --partition={slurm.partition}
#SBATCH --gres={slurm.gres}
#SBATCH --cpus-per-task={slurm.cpus_per_task}
#SBATCH --mem={slurm.mem}
#SBATCH --time={slurm.time}
#SBATCH --output={log_dir}/{config.experiment_name}_%j.out
#SBATCH --error={log_dir}/{config.experiment_name}_%j.err

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================"

# Environment variables
export PYTHONUNBUFFERED=1
export WANDB_MODE={config.logging.mode}

# Run training
python train.py --config {config_name}

echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"
"""

    # Write script
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make executable
    script_path.chmod(0o755)

    print(f"\n✅ SBATCH script generated: {script_path}")
    print(f"   Submit with: sbatch {script_path}")

    return str(script_path)


def submit_sbatch_script(script_path: str) -> bool:
    """
    Submit SBATCH script to SLURM.
    """
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=False,
        )

        print(result.stdout)

        if result.returncode == 0:
            print("✅ Job submitted successfully")
            return True
        else:
            print(f"❌ Job submission failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ Error: 'sbatch' command not found. Is SLURM installed?")
        return False
    except Exception as e:
        print(f"❌ Error submitting job: {e}")
        return False


def prompt_sbatch_submission(script_path: str) -> bool:
    """
    Prompt user to submit the generated SBATCH script.
    Automatically skips prompting if stdin is not available (non-interactive).
    """
    import sys

    # Check if stdin is available (interactive terminal)
    if not sys.stdin.isatty():
        print(
            "Non-interactive environment detected. Skipping SBATCH submission prompt."
        )
        print("To submit the job, run: sbatch " + script_path)
        return False

    try:
        response = input("\nSBATCH script generated. Submit to SLURM? (y/n): ")

        if response.lower() == "y":
            return submit_sbatch_script(script_path)
        else:
            print("Continuing with local interactive execution...")
            return False
    except EOFError:
        print("\nNo input available. Skipping SBATCH submission.")
        print("To submit the job, run: sbatch " + script_path)
        return False
