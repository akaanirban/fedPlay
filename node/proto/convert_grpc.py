#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:05:39 2020

@author: Anirban Das
"""

from grpc.tools import protoc
import sys

if len(sys.argv) > 1:
    protofile = sys.argv[1]
else:
    protofile = "./node.proto"

protoc.main(
    (
        '',
        '-I.',
        '--python_out=.',
        '--grpc_python_out=.',
        protofile,
    )
)