# ==============================================================================
# T-MoE Training — ECR Repository for Training Container
# ==============================================================================
# Private ECR repository for the training Docker image.
# Lifecycle policy keeps only the last 5 images for cost control.
# ==============================================================================

resource "aws_ecr_repository" "training" {
  name                 = "${var.project_name}-${var.environment}-training"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "T-MoE Training Container"
    Description = "Docker images for GPU training jobs"
  }
}

# --- Lifecycle Policy: Keep last 5 images ---
resource "aws_ecr_lifecycle_policy" "training" {
  repository = aws_ecr_repository.training.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only the last 5 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
