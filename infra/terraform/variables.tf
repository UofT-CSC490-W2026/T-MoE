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

variable "batch_instance_types" {
  description = "Instance types for Batch compute environment"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge"]
}

variable "batch_max_vcpus" {
  description = "Maximum vCPUs for Batch compute environment (cost control)"
  type        = number
  default     = 16
}

variable "batch_min_vcpus" {
  description = "Minimum vCPUs for Batch compute environment (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "batch_job_vcpus" {
  description = "vCPUs per training job"
  type        = number
  default     = 4
}

variable "batch_job_memory" {
  description = "Memory (MiB) per training job"
  type        = number
  default     = 12000
}
