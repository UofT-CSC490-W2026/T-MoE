resource "aws_launch_template" "batch_gpu" {
  name_prefix = "${var.project_name}-${var.environment}-batch-gpu-"

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

resource "aws_batch_compute_environment" "gpu_spot" {
  compute_environment_name = "${var.project_name}-${var.environment}-gpu-v1"
  type                     = "MANAGED"
  state                    = "ENABLED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = var.batch_min_vcpus
    max_vcpus           = var.batch_max_vcpus
    instance_type       = var.batch_instance_types
    subnets             = aws_subnet.batch_public[*].id
    security_group_ids  = [aws_security_group.batch.id]
    instance_role       = aws_iam_instance_profile.batch_ecs.arn

    ec2_configuration {
      image_type = "ECS_AL2023_NVIDIA"
    }

    launch_template {
      launch_template_id = aws_launch_template.batch_gpu.id
      version            = "$Latest"
    }

    tags = {
      Name = "${var.project_name}-${var.environment}-batch-instance"
    }
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-gpu-on-demand-ce"
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [
    aws_iam_role.batch_service,
    aws_iam_role_policy_attachment.batch_service,
    aws_iam_role_policy.batch_service_extras
  ]
}

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

  depends_on = [aws_batch_compute_environment.gpu_spot]
}

resource "aws_cloudwatch_log_group" "batch_training" {
  name              = "/aws/batch/${var.project_name}-${var.environment}/training"
  retention_in_days = 14

  tags = {
    Name        = "Batch Training Logs"
    Description = "Log retention for SPAR AWS Batch training jobs"
  }
}

resource "aws_batch_job_definition" "training" {
  name = "${var.project_name}-${var.environment}-training"
  type = "container"

  platform_capabilities = ["EC2"]

  retry_strategy {
    attempts = 2
  }

  timeout {
    attempt_duration_seconds = 43200
  }

  container_properties = jsonencode({
    image   = "${aws_ecr_repository.training.repository_url}:latest"
    command = ["--config", "gptneo_125m_lora"]

    resourceRequirements = [
      {
        type  = "VCPU"
        value = tostring(var.batch_job_vcpus)
      },
      {
        type  = "MEMORY"
        value = tostring(var.batch_job_memory)
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

resource "aws_iam_role" "batch_service" {
  name = "${var.project_name}-${var.environment}-batch-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowBatchAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "batch.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name        = "Batch Service Role"
    Description = "Service role for AWS Batch compute environment management"
  }
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role_policy" "batch_service_extras" {
  name = "${var.project_name}-${var.environment}-batch-service-extras"
  role = aws_iam_role.batch_service.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowECSListClusters"
        Effect = "Allow"
        Action = [
          "ecs:ListClusters",
          "ecs:DescribeClusters",
          "ecs:CreateCluster",
          "ecs:DeleteCluster"
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowBatchLogsManagement"
        Effect = "Allow"
        Action = [
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role" "batch_spot_fleet" {
  name = "${var.project_name}-${var.environment}-batch-spot-fleet"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowSpotFleetAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "spotfleet.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name        = "Batch Spot Fleet Role"
    Description = "Role for Spot Fleet instance management"
  }
}

resource "aws_iam_role_policy_attachment" "batch_spot_fleet" {
  role       = aws_iam_role.batch_spot_fleet.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

resource "aws_iam_role" "batch_ecs_task" {
  name = "${var.project_name}-${var.environment}-batch-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowECSTaskAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name        = "Batch ECS Task Role"
    Description = "Role assumed by training containers for S3 and CloudWatch access"
  }
}

resource "aws_iam_role_policy" "batch_ecs_s3" {
  name = "${var.project_name}-${var.environment}-batch-ecs-s3"
  role = aws_iam_role.batch_ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBucket"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [aws_s3_bucket.raw_data.arn]
      },
      {
        Sid    = "ReadWriteObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = ["${aws_s3_bucket.raw_data.arn}/*"]
      }
    ]
  })
}

resource "aws_iam_role_policy" "batch_ecs_logs" {
  name = "${var.project_name}-${var.environment}-batch-ecs-logs"
  role = aws_iam_role.batch_ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents"
        ]
        Resource = ["arn:aws:logs:${var.aws_region}:*:log-group:/aws/batch/*"]
      }
    ]
  })
}

resource "aws_iam_role_policy" "batch_ecs_ecr" {
  name = "${var.project_name}-${var.environment}-batch-ecs-ecr"
  role = aws_iam_role.batch_ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:GetAuthorizationToken"
        ]
        Resource = ["*"]
      }
    ]
  })
}

resource "aws_iam_role" "batch_ecs_instance" {
  name = "${var.project_name}-${var.environment}-batch-ecs-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEC2AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "Batch ECS Instance Role"
  }
}

resource "aws_iam_role_policy_attachment" "batch_ecs_instance" {
  role       = aws_iam_role.batch_ecs_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_ecs" {
  name = "${var.project_name}-${var.environment}-batch-ecs"
  role = aws_iam_role.batch_ecs_instance.name
}
