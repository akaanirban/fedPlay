### Deployment on AWS systems
To deploy the client nodes on AWS EC2 instances, we need 3 things: 
1. Terraform scripts: These will allow us to provision the EC2 instance, select the types, as well as create the security groups with necessary outbound and inbound ports.
    These are the `00-aws-infra.tf` for AWS and `01-gce-infra.tf` for GCE. We don't use GCE but if required, the GCE terraform file can be modified accordingly. 
    For security group inbound rule, we need to expose the ports our client docker containers will use. Here I expose ports between 5000 to 5100, as well as some other ports if we use docker swarm mode. 
    The later is absolutely not necessary for the purpose.
    
   ```hcl-terraform
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
    ```

2. Terraform variable script: This is where we mention the specific default values of the variables. This is the `variables.tf` script. We don't use a `tfvars` as the deployment is quite simple.
3. We need `ansible` scripts which will allow us to install docker, docker-compose on our EC2 nodes, `scp` our project folder and then start the client nodes by calling `docker-compose up --build`
   We first install docker , docker compose etc. Then we use `synchronize` to `scp` the `fedplay` directory directly to the worker nodes. We then run `docker compose` using the `docker-compose.yml` file in
   the project root to start the clients inside docker containers.
```yaml
        - name: Install docker-compose
          hosts: fedplay-workers
          gather_facts: yes
          tasks:
            - apt:
                name: docker-compose
                state: present
                update_cache: yes
    
        - name: SCP this current project folder into all other remote hosts
          hosts: fedplay-workers
          gather_facts: yes
          tasks:
            - synchronize: # use synchronize instead of copy as it is faster https://stackoverflow.com/a/27995384/8853476
                src: /home/anirban/Softwares/GitHub/fedPlay
                dest: /home/ubuntu/
        
        - name: run the service defined in docker-compose.yml
          hosts: fedplay-workers
          gather_facts: yes
          tasks:
            - docker_compose:
                project_src: /home/ubuntu/fedPlay
```

On successful application of the terraform scripts, we instruct terraform to create an `inventory` file containing the node names and public ip addresses via `02-create-inv.tf`. This will be the input to the `ansible` commands.
```
[swarm-master]
100.25.111.33 ansible_ssh_user=ubuntu
[swarm-nodes]
54.173.88.147 ansible_ssh_user=ubuntu
184.72.150.247 ansible_ssh_user=ubuntu
```
#### Deploy on AWS
```bash
    >> terraform plan
    >> terraform apply 
```
#### Orchestrate stuff using Ansible
```bash
    >> ./deploy-ansible.sh # orchestrates using ansible
    >> ./destroy-ansible.sh # destroys the ansible infrastructure. Basically stops the running docker containers.
```


Inspired from https://rsmitty.github.io/Multi-Cloud-Swarm/

How to authenticate ansible to do stuff on AWS EC2 instance with a ssh key: https://www.reddit.com/r/ansible/comments/b417g8/how_do_you_authenticate_to_aws_ec2_instance_with/