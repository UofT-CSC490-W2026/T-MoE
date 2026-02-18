# ==============================================================================
# T-MoE Data Ingestion - IAM Roles and Policies
# ==============================================================================
# Least-privilege IAM role for SageMaker processing jobs.
# Permissions scoped to specific S3 bucket and CloudWatch log groups.
# ==============================================================================

# --- SageMaker Execution Role ---
resource "aws_iam_role" "sagemaker_execution" {
  name = "${var.project_name}-${var.environment}-sagemaker-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowSageMakerAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name        = "SageMaker Execution Role"
    Description = "Execution role for T-MoE SageMaker data ingestion processing jobs"
  }
}

# --- S3 Access Policy (Scoped to Raw Data Bucket Only) ---
resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "${var.project_name}-${var.environment}-s3-access"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListRawDataBucket"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.raw_data.arn
        ]
      },
      {
        Sid    = "ReadWriteRawDataObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.raw_data.arn}/*"
        ]
      }
    ]
  })
}

# --- CloudWatch Logs Policy ---
resource "aws_iam_role_policy" "sagemaker_logs" {
  name = "${var.project_name}-${var.environment}-cloudwatch-logs"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogsAccess"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents"
        ]
        Resource = [
          "arn:aws:logs:${var.aws_region}:*:log-group:/aws/sagemaker/*"
        ]
      }
    ]
  })
}

# --- ECR Read Access (for pulling SageMaker HuggingFace container images) ---
resource "aws_iam_role_policy_attachment" "sagemaker_ecr_readonly" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# ==============================================================================
# AWS Batch — IAM Roles
# ==============================================================================

# --- Batch Service Role ---
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

# --- Missing Permissions for Batch Service (Required for cluster management/deletion) ---
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

# --- Spot Fleet Role ---
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

# --- ECS Task Role (used by Batch containers) ---
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

# --- ECS Task: S3 Access (scoped to raw data bucket) ---
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

# --- ECS Task: CloudWatch Logs ---
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

# --- ECS Task: ECR Pull ---
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

# --- ECS Instance Profile (for Batch compute instances) ---
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
