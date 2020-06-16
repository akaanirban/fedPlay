##General vars
variable "ssh_user" {
  default = "ubuntu"
}
variable "public_key_path" {
  default = "/homa/anirban/.ssh/id_rsa.pub"
}
variable "private_key_path" {
  default = "/homa/anirban/.ssh/id_rsa"
}
##AWS Specific Vars
variable "aws_worker_count" {
  default = 3
}
variable "aws_key_name" {
  default = "terraform-keys"
}
variable "aws_instance_size" {
  default = "t2.micro"
}
variable "aws_region" {
  default = "us-east-1"
}
variable "aws_credentials_path" {
  default = "/homa/anirban/.aws/credentials"
}
# ##GCE Specific Vars
# variable "gce_worker_count" {
#   default = 1
# }
# variable "gce_creds_path" {
#   default = "/homa/anirban/gce-creds.json"
# }
# variable "gce_project" {
#   default = "test-project"
# }
# variable "gce_region" {
#   default = "us-central1"
# }
# variable "gce_instance_size" {
#   default = "n1-standard-1"
# }