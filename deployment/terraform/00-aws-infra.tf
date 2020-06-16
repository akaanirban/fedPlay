##Amazon Infrastructure
provider "aws" {
  region = var.aws_region
  shared_credentials_file = var.aws_credentials_path
}

##Create fedplay security group
resource "aws_security_group" "fedplay" {
  name        = "fedplay"
  description = "Allow all inbound traffic necessary"
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 2377
    to_port     = 2377
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 7946
    to_port     = 7946
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 7946
    to_port     = 7946
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 4789
    to_port     = 4789
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  # Ingress ports to expose inbound ports of simulated clients on a VM
  ingress {
    from_port   = 5000
    to_port     = 5100
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    cidr_blocks = [
      "0.0.0.0/0",
    ]
  }
  tags = {
    Name = "fedplay"
  }
}

##Find latest Ubuntu 16.04 AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-bionic-18.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

##Create fedplay worker instances.
resource "aws_instance" "fedplay-workers" {
  depends_on             = ["aws_security_group.fedplay"]
  ami                    = data.aws_ami.ubuntu.id
  instance_type = var.aws_instance_size
  vpc_security_group_ids = [aws_security_group.fedplay.id]
  key_name               = var.aws_key_name
  count                  = var.aws_worker_count
  tags = {
    Name = "fedplay-worker-${count.index}"
  }
}

resource "aws_key_pair" "terraform-keys" {
  key_name = "terraform-keys"
  public_key = file("./terraform-keys.pub")
}