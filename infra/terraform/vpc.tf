data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "batch" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-${var.environment}-batch-vpc"
  }
}

resource "aws_subnet" "batch_public" {
  count = 2

  vpc_id                  = aws_vpc.batch.id
  cidr_block              = cidrsubnet(aws_vpc.batch.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-${var.environment}-batch-public-${count.index}"
  }
}

resource "aws_internet_gateway" "batch" {
  vpc_id = aws_vpc.batch.id

  tags = {
    Name = "${var.project_name}-${var.environment}-batch-igw"
  }
}

resource "aws_route_table" "batch_public" {
  vpc_id = aws_vpc.batch.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.batch.id
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-batch-public-rt"
  }
}

resource "aws_route_table_association" "batch_public" {
  count = 2

  subnet_id      = aws_subnet.batch_public[count.index].id
  route_table_id = aws_route_table.batch_public.id
}

resource "aws_security_group" "batch" {
  name_prefix = "${var.project_name}-${var.environment}-batch-"
  description = "Security group for Batch GPU training instances"
  vpc_id      = aws_vpc.batch.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-batch-sg"
  }

  lifecycle {
    create_before_destroy = true
  }
}
