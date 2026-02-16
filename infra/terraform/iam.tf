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
