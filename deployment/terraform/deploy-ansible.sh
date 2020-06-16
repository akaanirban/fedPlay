#!/bin/bash

ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -b -i ansible-inventory ansible-docker-compose.yml --private-key=terraform-keys.pem
