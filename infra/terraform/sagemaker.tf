# ==============================================================================
# T-MoE Data Ingestion - SageMaker Configuration
# ==============================================================================
# CloudWatch log group and future SageMaker resource placeholders.
#
# NOTE: Processing jobs are launched programmatically via run_processing.py,
# not defined in Terraform. This file establishes conventions and provides
# hooks for future SageMaker Pipelines / Studio integration.
# ==============================================================================

# --- CloudWatch Log Group for SageMaker Processing Jobs ---
resource "aws_cloudwatch_log_group" "sagemaker_processing" {
  name              = "/aws/sagemaker/processing-jobs/${var.project_name}-${var.environment}"
  retention_in_days = 14

  tags = {
    Name        = "SageMaker Processing Job Logs"
    Description = "Log retention for T-MoE data ingestion processing jobs"
  }
}

# ==============================================================================
# FUTURE: SageMaker Studio Domain
# ==============================================================================
# Uncomment and configure when SageMaker Studio is needed for interactive
# development and experiment tracking.
#
# resource "aws_sagemaker_domain" "tmoe_studio" {
#   domain_name = "${var.project_name}-${var.environment}-studio"
#   auth_mode   = "IAM"
#   vpc_id      = var.vpc_id
#   subnet_ids  = var.subnet_ids
#
#   default_user_settings {
#     execution_role = aws_iam_role.sagemaker_execution.arn
#   }
#
#   tags = {
#     Name = "T-MoE SageMaker Studio"
#   }
# }

# ==============================================================================
# FUTURE: SageMaker Pipeline Definition
# ==============================================================================
# When ready to automate ingestion → preprocessing → training:
#
# resource "aws_sagemaker_pipeline" "tmoe_ingestion" {
#   pipeline_name         = "${var.project_name}-${var.environment}-ingestion"
#   pipeline_display_name = "T-MoE Data Ingestion Pipeline"
#   role_arn              = aws_iam_role.sagemaker_execution.arn
#
#   pipeline_definition = jsonencode({
#     // Step Functions or SageMaker Pipeline JSON definition
#   })
# }

# ==============================================================================
# FUTURE: EventBridge Scheduled Rule for Periodic Ingestion
# ==============================================================================
# resource "aws_cloudwatch_event_rule" "periodic_ingestion" {
#   name                = "${var.project_name}-${var.environment}-periodic-ingestion"
#   description         = "Trigger data ingestion on schedule"
#   schedule_expression = "rate(7 days)"
# }
