# T-MoE Infrastructure — Data Ingestion Pipeline

Production-ready AWS SageMaker data ingestion pipeline for the T-MoE project. Uses Terraform for IaC and SageMaker HuggingFaceProcessor to ingest HuggingFace datasets into an S3 data lake.

## Architecture

### SageMaker Mode (use_sagemaker: true)
```
┌──────────────────┐     ┌─────────────────────────┐     ┌──────────────────┐
│  run_pipeline.py │────▶│  SageMaker Processing   │────▶│   S3 Raw Data    │
│  (Local / CI/CD)  │     │  (HuggingFace Container)│     │   Landing Zone   │
└──────────────────┘     └─────────────────────────┘     └──────────────────┘
        │                           │                            │
        ▼                           ▼                            ▼
  config.yaml              processing_script.py          datasets/raw/
  .env / TF outputs        - Load from HF Hub            ├── train.jsonl
                           - Validate splits             ├── validation.jsonl
                           - Write to /opt/ml/output     ├── test.jsonl
                                                         └── metadata.json
```

### Fallback Mode (use_sagemaker: false)
```
┌──────────────────┐     ┌─────────────────────────┐     ┌──────────────────┐
│  run_pipeline.py │────▶│  fallback_ingestion.py  │────▶│   S3 Raw Data    │
│  (Local / CI/CD)  │     │  (Direct boto3 upload)  │     │   Landing Zone   │
└──────────────────┘     └─────────────────────────┘     └──────────────────┘
        │                           │                            │
        ▼                           ▼                            ▼
  config.yaml              - Load from HF Hub            datasets/raw/
  use_sagemaker: false     - Write to /tmp              ├── train.jsonl
                           - Upload to S3                ├── validation.jsonl
                                                         ├── test.jsonl
                                                         └── metadata.json
```

## Directory Structure

```
infra/
├── terraform/                 # Infrastructure as Code
│   ├── main.tf                # Provider configuration
│   ├── variables.tf           # Input variables
│   ├── backend.tf             # State management
│   ├── s3.tf                  # S3 bucket (encrypted, versioned)
│   ├── iam.tf                 # IAM roles (least privilege)
│   ├── sagemaker.tf           # CloudWatch log groups
│   └── outputs.tf             # Infrastructure output values
├── data_ingestion/            # Data ingestion modules
│   ├── processing_script.py   # Runs inside SageMaker container
│   ├── run_processing.py      # Launches SageMaker processing job
│   ├── fallback_ingestion.py  # Direct S3 upload (SageMaker-independent)
│   ├── logger_utils.py        # Structured logging
│   └── requirements.txt       # Python dependencies
├── s3client/                  # S3 client utilities
│   ├── __init__.py
│   └── client.py              # Upload/download/list operations
├── config/
│   └── config.py              # Centralized config management
└── README.md                  # This file
```

## Prerequisites

- **AWS Account** with permissions to create S3, IAM, SageMaker, CloudWatch resources
- **AWS CLI** v2+ configured (`aws configure`)
- **Terraform** >= 1.5.0
- **Python** >= 3.10
- IAM user/role with permissions to run `terraform apply`

## Quick Start

### 1. Deploy Infrastructure

```bash
cd infra/terraform

# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Deploy resources
terraform apply
```

### 2. Configure Environment

```bash
# Copy .env template
cp ../../.env.example ../../.env

# Populate from Terraform outputs
terraform output raw_data_bucket_name    # → RAW_DATA_BUCKET
terraform output sagemaker_execution_role_arn  # → SAGEMAKER_ROLE_ARN (only needed for SageMaker mode)

# Or use the auto-generated snippet:
terraform output env_configuration
```

### 3. Install Dependencies

```bash
# From project root
pip install -r infra/data_ingestion/requirements.txt
```

### 4. Run Data Ingestion

#### Option A: Fallback Mode (Direct S3 Upload — No SageMaker)

```bash
# Ensure use_sagemaker: false in config.yaml (default)
python run_pipeline.py
```

**When to use**: SageMaker unavailable, local development, cost optimization

**Requirements**: AWS credentials, S3 bucket access, HuggingFace Hub access

#### Option B: SageMaker Mode

```bash
# Set use_sagemaker: true in config.yaml or via environment
USE_SAGEMAKER=true python run_pipeline.py

# Or update config.yaml:
# data_ingestion:
#   use_sagemaker: true
python run_pipeline.py
```

**When to use**: Production workloads, large datasets, managed infrastructure

**Requirements**: SageMaker execution role, all of Option A requirements

## Security

| Feature | Status |
|---------|--------|
| No hardcoded credentials | ✅ IAM roles only |
| S3 encryption at rest | ✅ AES256 |
| S3 public access blocked | ✅ All 4 settings |
| IAM least privilege | ✅ Scoped to specific bucket |
| Terraform state encryption | ✅ When using S3 backend |
| CloudWatch audit logging | ✅ Processing job logs |

## Terraform Remote State (Optional)

For production use, migrate from local to S3 backend:

```bash
# 1. Create state bucket
aws s3 mb s3://tmoe-terraform-state --region us-east-1

# 2. Enable versioning
aws s3api put-bucket-versioning \
  --bucket tmoe-terraform-state \
  --versioning-configuration Status=Enabled

# 3. Create DynamoDB lock table
aws dynamodb create-table \
  --table-name tmoe-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# 4. Edit backend.tf: uncomment S3 block, comment local block

# 5. Migrate state
terraform init -migrate-state
```

## Cost Optimization

- **Spot Instances**: Set `use_spot_instances: true` in config.yaml (~70% savings)
- **Lifecycle Policies**: Old data auto-transitions to Glacier after 90 days
- **Instance Right-Sizing**: Use `ml.m5.large` for smaller datasets
- **Budget Alerts**: Set up AWS Budget alerts for SageMaker spend

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `NoCredentialsError` | Run `aws configure` or check IAM role |
| `AccessDenied` on S3 | Check IAM policy in `iam.tf` |
| Processing job timeout | Increase `max_runtime_seconds` in config |
| Terraform state lock | Run `terraform force-unlock <LOCK_ID>` |
| Missing HF dataset | Verify dataset name on huggingface.co |
