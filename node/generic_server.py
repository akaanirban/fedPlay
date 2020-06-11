#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 01:20:30 2020

@author: Anirban Das
"""

from abc import ABC, abstractmethod


class Node(ABC):
    """
    Abstract class for a Node
    """

    def __init__(self, config, isHub):
        self.config = config
        self.isHub = isHub
        super().__init__()

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def initializeNetwork(self):
        pass

    @abstractmethod
    def initializeData(self):
        pass

    @abstractmethod
    def sendModel(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def aggregateClientModels(self):
        pass

    @abstractmethod
    def clientSelection(self):
        pass

    @abstractmethod
    def queryModelFromClients(self):
        pass