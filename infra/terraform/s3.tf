# ==============================================================================
# T-MoE Data Ingestion - S3 Raw Data Landing Bucket
# ==============================================================================
# Secure S3 bucket for HuggingFace dataset ingestion.
# Features: encryption, versioning, public access blocked, lifecycle policies.
# ==============================================================================

# Random suffix for globally unique bucket names
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# --- Raw Data Landing Bucket ---
resource "aws_s3_bucket" "raw_data" {
  bucket = "${var.project_name}-${var.environment}-${var.raw_bucket_prefix}-${random_id.bucket_suffix.hex}"

  # Allow deletion even if non-empty (required for dev environments with managed data)
  force_destroy = true

  tags = {
    Name        = "T-MoE Raw Data Landing Zone"
    DataStage   = "raw"
    Description = "HuggingFace dataset ingestion landing zone"
  }
}

# --- Versioning ---
resource "aws_s3_bucket_versioning" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

# --- Server-Side Encryption (AES256) ---
resource "aws_s3_bucket_server_side_encryption_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# --- Block ALL Public Access ---
resource "aws_s3_bucket_public_access_block" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Lifecycle Policy: Transition to Glacier for Cost Optimization ---
resource "aws_s3_bucket_lifecycle_configuration" "raw_data" {
  count  = var.enable_lifecycle_policy ? 1 : 0
  bucket = aws_s3_bucket.raw_data.id

  rule {
    id     = "archive-old-raw-data"
    status = "Enabled"

    filter {
      prefix = "datasets/"
    }

    transition {
      days          = var.lifecycle_glacier_days
      storage_class = "GLACIER"
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}
