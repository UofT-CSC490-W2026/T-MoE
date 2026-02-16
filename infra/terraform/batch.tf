# ==============================================================================
# T-MoE Training — AWS Batch Compute Environment, Job Queue, Job Definition
# ==============================================================================
# GPU-capable Spot compute environment that scales to 0 when idle.
# Job definition runs the training pipeline Docker container.
# ==============================================================================

# --- Launch Template (ECS-optimized GPU AMI) ---
resource "aws_launch_template" "batch_gpu" {
  name_prefix = "${var.project_name}-${var.environment}-batch-gpu-"

  # Attach additional storage for model weights and datasets
  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-batch-gpu-lt"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# --- Batch Compute Environment (GPU Spot Instances) ---
resource "aws_batch_compute_environment" "gpu_spot" {
  compute_environment_name = "${var.project_name}-${var.environment}-gpu-spot"
  type                     = "MANAGED"
  state                    = "ENABLED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "SPOT"
    bid_percentage      = 100 # Up to On-Demand price
    spot_iam_fleet_role = aws_iam_role.batch_spot_fleet.arn
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"

    min_vcpus = var.batch_min_vcpus
    max_vcpus = var.batch_max_vcpus

    instance_type = var.batch_instance_types

    subnets            = aws_subnet.batch_public[*].id
    security_group_ids = [aws_security_group.batch.id]

    instance_role = aws_iam_instance_profile.batch_ecs.arn

    launch_template {
      launch_template_id = aws_launch_template.batch_gpu.id
      version            = "$Latest"
    }

    tags = {
      Name = "${var.project_name}-${var.environment}-batch-instance"
    }
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-gpu-spot-ce"
  }

  depends_on = [aws_iam_role.batch_service]
}

# --- Job Queue ---
resource "aws_batch_job_queue" "training" {
  name     = "${var.project_name}-${var.environment}-training"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_spot.arn
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-training-queue"
  }
}

# --- CloudWatch Log Group for Batch Jobs ---
resource "aws_cloudwatch_log_group" "batch_training" {
  name              = "/aws/batch/${var.project_name}-${var.environment}/training"
  retention_in_days = 14

  tags = {
    Name        = "Batch Training Logs"
    Description = "Log retention for T-MoE AWS Batch training jobs"
  }
}

# --- Job Definition ---
resource "aws_batch_job_definition" "training" {
  name = "${var.project_name}-${var.environment}-training"
  type = "container"

  platform_capabilities = ["EC2"]

  retry_strategy {
    attempts = 2
  }

  timeout {
    attempt_duration_seconds = 43200 # 12 hours max
  }

  container_properties = jsonencode({
    image   = "${aws_ecr_repository.training.repository_url}:latest"
    command = ["python", "run_training_pipeline.py", "--mode", "container"]

    resourceRequirements = [
      {
        type  = "VCPU"
        value = tostring(var.batch_job_vcpus)
      },
      {
        type  = "MEMORY"
        value = tostring(var.batch_job_memory)
      },
      {
        type  = "GPU"
        value = "1"
      }
    ]

    executionRoleArn = aws_iam_role.batch_ecs_task.arn
    jobRoleArn       = aws_iam_role.batch_ecs_task.arn

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch_training.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "training"
      }
    }

    environment = [
      {
        name  = "RAW_DATA_BUCKET"
        value = aws_s3_bucket.raw_data.id
      },
      {
        name  = "AWS_REGION"
        value = var.aws_region
      },
      {
        name  = "ENVIRONMENT"
        value = var.environment
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-${var.environment}-training-job-def"
  }
}
