#!/bin/bash

ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -b -i ansible-inventory docker-compose-destroy.yml --private-key=terraform-keys.pem
