#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/15/20 9:54 PM 2020

@author: Anirban Das
"""
import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

import fedplay as play

client = play.HFLClient()
client.start(port=8080)
