#!/bin/bash
# ==============================================================================
# T-MoE Disaster Recovery - Restoration Script
# ==============================================================================
# This script automates Phase 4 of the Disaster Recovery Demo.
# It applies Terraform, restores secrets to the .env file, and pushes the Docker image.
# ==============================================================================

set -e

echo "==========================================================="
echo "🚧 T-MoE DISASTER RECOVERY RESTORATION 🚧"
echo "==========================================================="

# 1. Restore Infrastructure via Terraform
echo -e "\n[1/3] Restoring Infrastructure via Terraform..."
cd infra/terraform
terraform apply -auto-approve
cd ../..

# 2. Restore Secrets to .env
# The local_file resource in Terraform regenerates .env with AWS configs.
# This step appends our local API keys back into the file.
echo -e "\n[2/3] Restoring Secrets to .env..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY=$WANDB_API_KEY" >> .env
    echo "✅ WANDB_API_KEY successfully restored to .env"a




    
else
    echo "⚠️  WARNING: WANDB_API_KEY is not set in your current terminal session!"
    echo "    Please add it manually to .env or run: export WANDB_API_KEY='...'"
fi

# 3. Restore Data Processing Services (Docker)
echo -e "\n[3/3] Restoring Data Processing Services (ECR)..."
source .env # Export variables needed for the docker script
./scripts/docker_build_push.sh

echo "==========================================================="
echo "✅ RESTORATION COMPLETE! Ready to run verification."
echo "==========================================================="
