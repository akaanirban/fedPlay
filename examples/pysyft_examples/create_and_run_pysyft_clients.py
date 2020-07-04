#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/16/20 2:54 PM 2020

@author: Anirban Das
"""

import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, path)
import fedplay as play
import yaml
import random
import os
import subprocess
import argparse
import uuid

random.seed(0)


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Deployment details of the PySyft Websocket Workers')
    parser.add_argument('--num', type=int, nargs='?', default=1,
                        help='Number of workers')
    parser.add_argument('--start', type=int, nargs='?', default=5000,
                        help='Starting port for the first worker')
    parser.add_argument('--cpu', type=float, nargs='?', default=0.2,
                        help='CPU to be allocated to each worker container. CPU value v. 0<v<4')
    parser.add_argument('--mem', type=float, nargs='?', default=512,
                        help='Memory to be allocated to each worker container. Memory value m in megabytes. 0< m < 32*1024')
    parser.add_argument('--seed', type=int, nargs='?', default=42,
                        help='Seed to be used.')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    # Parse input arguments
    args = parse_args()
    yaml = yaml.dump(play.CreatePySyftClients(args.num,
                                              args.start,
                                              args.cpu,
                                              args.mem,
                                              args.seed))
    print(yaml)
    filename = f"docker-compose-{uuid.uuid4()}.yaml"
    with open(filename, "w") as f:
        f.write(yaml)
    # deploy docker containers
    try:
        cmd = f"docker-compose -f {filename} up --remove-orphans"
        subprocess.call(cmd, shell=True)
    except Exception as e:
        print("Gracefully shutting down docker containers")
        cmd = f"docker-compose -f {filename} down"
        subprocess.call(cmd, shell=True)
    finally:
        print("Deleting docker compose files")
        os.remove(filename)
