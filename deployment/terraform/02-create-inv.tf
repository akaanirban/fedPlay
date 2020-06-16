resource "null_resource" "ansible-provision" {
  depends_on = ["aws_instance.fedplay-workers"]

  provisioner "local-exec" {
    command = "echo \"[fedplay-workers]\" >> ansible-inventory"
  }

  provisioner "local-exec" {
    command = "echo \"${join("\n",formatlist("%s ansible_ssh_user=%s", aws_instance.fedplay-workers.*.public_ip, var.ssh_user))}\" >> ansible-inventory"
  }

}
