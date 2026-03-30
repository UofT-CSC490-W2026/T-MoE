#!/usr/bin/env bash
# ==============================================================================
# SPAR — Build and Push Training Docker Image to ECR
# ==============================================================================
#
# Usage:
#   ./scripts/docker_build_push.sh
#   ./scripts/docker_build_push.sh --tag v1.0
#
# Prerequisites:
#   - AWS CLI configured with ECR access
#   - Docker installed and running
#   - Terraform applied (to create ECR repo)
#
# Environment Variables (auto-detected from Terraform or set manually):
#   ECR_REPOSITORY_URL  — Full ECR repository URL
#   AWS_REGION          — AWS region (default: ca-central-1)
# ==============================================================================

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TERRAFORM_DIR="${PROJECT_ROOT}/infra/terraform"

# Parse arguments
TAG="latest"
while [[ $# -gt 0 ]]; do
  case $1 in
    --tag)
      TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# --- Resolve ECR URL ---
if [ -z "${ECR_REPOSITORY_URL:-}" ]; then
  echo "🔍 ECR_REPOSITORY_URL not set, reading from Terraform..."
  if [ -f "${TERRAFORM_DIR}/terraform.tfstate" ]; then
    ECR_REPOSITORY_URL=$(cd "${TERRAFORM_DIR}" && terraform output -raw ecr_repository_url 2>/dev/null || true)
  fi

  if [ -z "${ECR_REPOSITORY_URL:-}" ]; then
    echo "❌ Cannot determine ECR_REPOSITORY_URL."
    echo "   Set it manually: export ECR_REPOSITORY_URL=<url>"
    echo "   Or run: cd infra/terraform && terraform output ecr_repository_url"
    exit 1
  fi
fi

# --- Resolve AWS Region ---
if [ -z "${AWS_REGION:-}" ]; then
  AWS_REGION=$(cd "${TERRAFORM_DIR}" && terraform output -raw aws_region 2>/dev/null || echo "ca-central-1")
fi

# Extract the registry URL (everything before the first /)
ECR_REGISTRY="${ECR_REPOSITORY_URL%%/*}"

# Get git SHA for tagging (short hash)
GIT_SHA=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "========================================"
echo "SPAR Training — Docker Build & Push"
echo "========================================"
echo "  Project Root : ${PROJECT_ROOT}"
echo "  ECR URL      : ${ECR_REPOSITORY_URL}"
echo "  AWS Region   : ${AWS_REGION}"
echo "  Tag          : ${TAG}"
echo "  Git SHA      : ${GIT_SHA}"
echo "========================================"

# --- Step 1: Authenticate with ECR ---
echo ""
echo "🔑 Authenticating with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# --- Step 2: Build image ---
echo ""
echo "🔨 Building Docker image..."
docker build \
  --platform linux/amd64 \
  --tag "${ECR_REPOSITORY_URL}:${TAG}" \
  --tag "${ECR_REPOSITORY_URL}:${GIT_SHA}" \
  --file "${PROJECT_ROOT}/Dockerfile" \
  "${PROJECT_ROOT}"

# --- Step 3: Push to ECR ---
echo ""
echo "📤 Pushing image to ECR..."
docker push "${ECR_REPOSITORY_URL}:${TAG}"
docker push "${ECR_REPOSITORY_URL}:${GIT_SHA}"

echo ""
echo "========================================"
echo "✅ Image pushed successfully!"
echo "   ${ECR_REPOSITORY_URL}:${TAG}"
echo "   ${ECR_REPOSITORY_URL}:${GIT_SHA}"
echo "========================================"
