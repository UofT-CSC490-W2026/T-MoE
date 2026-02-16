# ==============================================================================
# T-MoE Data Ingestion - Terraform Backend Configuration
# ==============================================================================
#
# REMOTE STATE SETUP (when ready):
#
# 1. Create S3 bucket for state:
#    aws s3 mb s3://tmoe-terraform-state --region us-east-1
#
# 2. Enable versioning on state bucket:
#    aws s3api put-bucket-versioning \
#      --bucket tmoe-terraform-state \
#      --versioning-configuration Status=Enabled
#
# 3. Create DynamoDB table for state locking:
#    aws dynamodb create-table \
#      --table-name tmoe-terraform-locks \
#      --attribute-definitions AttributeName=LockID,AttributeType=S \
#      --key-schema AttributeName=LockID,KeyType=HASH \
#      --billing-mode PAY_PER_REQUEST \
#      --region us-east-1
#
# 4. Uncomment the S3 backend block below and comment the local block.
#
# 5. Run: terraform init -migrate-state
#
# ==============================================================================

# --- Remote S3 Backend (uncomment when ready) ---
# terraform {
#   backend "s3" {
#     bucket         = "tmoe-terraform-state"
#     key            = "data-ingestion/terraform.tfstate"
#     region         = "us-east-1"
#     encrypt        = true
#     dynamodb_table = "tmoe-terraform-locks"
#   }
# }

# --- Local Backend (default for initial development) ---
terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}
