# ==============================================================================
# T-MoE Data Ingestion - Terraform Variables
# ==============================================================================

variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "ca-central-1"
}

variable "project_name" {
  description = "Project name used in resource naming conventions"
  type        = string
  default     = "tmoe"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "enable_versioning" {
  description = "Enable S3 bucket versioning for data protection"
  type        = bool
  default     = true
}

variable "enable_lifecycle_policy" {
  description = "Enable S3 lifecycle policy to transition old data to Glacier"
  type        = bool
  default     = true
}

variable "lifecycle_glacier_days" {
  description = "Number of days after which to transition objects to Glacier"
  type        = number
  default     = 90
}

variable "raw_bucket_prefix" {
  description = "Prefix for the raw data S3 bucket name"
  type        = string
  default     = "raw-data"
}
