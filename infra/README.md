## Infrastructure Setup

This document covers the one-time setup of the cloud infrastructure required to run T-MoE.

## Prerequisites

- Python 3.11+
- AWS CLI
- Terraform
- Modal CLI (`pip install modal`)
- A HuggingFace account
- A WandB account

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. AWS Setup

### Configure AWS CLI

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and default region
```

### Provision Infrastructure with Terraform

```bash
cd infra/terraform
terraform init
terraform apply
```

After `terraform apply` completes, capture the S3 bucket name:

```bash
terraform output env_configuration
```

Set it as an environment variable:

```bash
export RAW_DATA_BUCKET=<your-bucket-name-from-terraform-output>
```

---

## 3. WandB Setup

```bash
wandb login
```

Set the following environment variable:

```bash
export WANDB_API_KEY=<your-wandb-api-key>
```

---

## 4. HuggingFace Setup

A HuggingFace access token is required to download models (e.g., GPT-Neo) and optionally gated datasets.

Generate a token at: https://huggingface.co/settings/tokens

```bash
export HF_TOKEN=<your-huggingface-token>
```

---

## 5. Modal Setup

### First-time setup

If you are logging in from a new machine or need to switch to a specific workspace profile (e.g., `dev-tmoe`):

```bash
python3 -m modal setup
```
Follow the browser prompts to authenticate. Once complete, your token is saved locally to `~/.modal.toml`.

### Create Environment and Workspace Secret

Create the `tmoe-secrets` secret in the `main` environment:

```bash
modal secret create tmoe-secrets \
    HF_TOKEN=your_hf_token \
    WANDB_API_KEY=your_wandb_key
```

Or via the Modal Dashboard:
1. Go to https://modal.com → Secrets → New Secret (Custom).
2. Make sure you are in the `main` environment.
3. Name it: `tmoe-secrets`.
4. Add the following key-value pairs:

| Key                     | Value                          | Required?    |
|-------------------------|--------------------------------|--------------|
| `HF_TOKEN`              | Your HuggingFace access token  | Yes          |
| `WANDB_API_KEY`         | Your WandB API key             | Yes          |
| `AWS_ACCESS_KEY_ID`     | Your AWS Access Key ID         | If using S3  |
| `AWS_SECRET_ACCESS_KEY` | Your AWS Secret Access Key     | If using S3  |

These secrets are automatically injected into Modal containers at runtime. You never hardcode them in code.

---

## 6. Verify Setup

```bash
# Verify AWS access
aws s3 ls s3://$RAW_DATA_BUCKET

# Verify Modal login
modal token show

# Verify WandB login
wandb status
```

---

## Environment Variables Summary

| Variable                | Source               | Used by              |
|-------------------------|----------------------|----------------------|
| `RAW_DATA_BUCKET`       | Terraform output     | `run_pipeline.py`    |
| `WANDB_API_KEY`         | WandB dashboard      | `scripts/train.py`   |
| `HF_TOKEN`              | HuggingFace          | `scripts/prepare_data.py` |
| `AWS_ACCESS_KEY_ID`     | AWS Console          | Terraform, boto3     |
| `AWS_SECRET_ACCESS_KEY` | AWS Console          | Terraform, boto3     |
