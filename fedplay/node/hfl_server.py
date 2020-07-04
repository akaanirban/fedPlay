#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 01:28:12 2020

@author: Anirban Das
"""
from random import random

import grpc
import concurrent.futures
import collections
import fedplay as play
from fedplay.serde import serialize, deserialize
from fedplay.node.client_rpc_utils_for_server import initializeClient, initializeDataClient, trainFunc, sendModel
from fedplay.node.generic_server import Node
from fedplay.node.utils.costs import linear_cost
from fedplay.node.proto import functions_pb2, functions_pb2_grpc
import logging
import sys
import copy
import numpy as np

logging.basicConfig(level=logging.WARNING)


class MLServer(Node):
    """
        A federated learning server | Also acts as a hub
        This server is for ML based approach. Not DL
        Further, this training procedure is horizontal federated learning
        Therefore, we do not need to send the Xw vector at each global round
        The changes are just in the train and sendModel method.
    """

    def __init__(self, config):
        logging.info(f"Booting Server/Hub {config.get('index')}")
        self.config = config
        self.client_config = config.get('client_config')
        self.X_train = config.get('X_train')
        self.y_train = config.get('y_train')
        self.datapoints_per_device = config.get('datapoints_per_device')
        self.coordinates_per_dc = config.get('coordinates_per_dc')
        self.initialize()
        self.client_connections = collections.OrderedDict()
        self.sampled_clients = collections.OrderedDict()
        self.client_models = collections.OrderedDict()
        self.loss = []
        self.client_list = []

    def initialize(self):
        self.alpha: float = self.config.get('alpha')
        self.lambduh: float = 0.01  # self.config.get('lambduh')
        self.local_epoch: int = 5
        self.global_Xtheta: np.array = np.zeros((self.X_train.shape[0], 1))  # self.config.get('Xtheta')
        self.global_model: np.array = np.zeros((self.X_train.shape[1], 1))  # self.config.get('theta')
        # self.costs = []
        # self.X = self.config.get('X')
        # self.y = self.config.get('y')
        # self.index = self.config.get('index')
        # self.offset = self.config.get('offset')
        # self.theta_average = self.theta[self.index * self.offset: (self.index + 1) * self.offset]
        # self.local_estimate = None
        # # From https://github.com/akaanirban/BlockCoordinateDescent/blob/c18622aeb8f08cba87d9ad47e4b1b252e93b213a/loss_functions.py#L78
        # self.lipschitz_constant = np.max(np.linalg.eig(self.X.T @ self.X)[0])
        # self.global_Xtheta = None
        # self.global_model = None

    def initializeNetwork(self):
        super().initializeNetwork()

    def initializeData(self):
        executor = concurrent.futures.ThreadPoolExecutor(len(self.client_list))
        futures = {executor.submit(initializeDataClient,
                                   [client.get("client_id"), client.get("channel"),
                                    self.X_train[
                                    client.get("client_sequence") * self.datapoints_per_device: (client.get(
                                        "client_sequence") + 1) * self.datapoints_per_device, :],
                                    self.y_train[
                                    client.get("client_sequence") * self.datapoints_per_device: (client.get(
                                        "client_sequence") + 1) * self.datapoints_per_device, :]]
                                   ): client.get("client_id") for client in self.client_list}
        results, _ = concurrent.futures.wait(futures)
        logging.info(
            f"Server/Hub {self.config.get('index')} finished data setup of clients: {[res.result() for res in results]}.")

    def calculateTrainingLoss(self):
        self.loss.append(linear_cost(self.global_model, self.X_train, self.y_train, self.lambduh))

    def train(self, clients_per_round=0):
        for global_round in range(self.config.get('T')):
            logging.info(
                f"Server/Hub {self.config.get('index')} starting global round {global_round}.")
            sampled_clients = self.clientSelection(clients_per_round)
            executor = concurrent.futures.ThreadPoolExecutor(len(sampled_clients))
            futures = {executor.submit(trainFunc,
                                       [client.get("client_id"), client.get("channel"),
                                        self.local_epoch, self.lambduh,
                                        self.global_Xtheta[client.get("client_sequence") * self.datapoints_per_device:
                                                           (client.get(
                                                               "client_sequence") + 1) * self.datapoints_per_device, :],
                                        "",
                                        ]): client.get("client_id") for client in self.client_list}
            results, b = concurrent.futures.wait(futures)
            clientModels = collections.OrderedDict()
            for result in results:
                # https://stackoverflow.com/a/52082992/8853476
                result = result.result()
                client_id = result[0]
                received_model = result[1]
                for clidx, client in enumerate(self.client_list):
                    if client.get("client_id") == client_id:
                        self.client_list[clidx]["model"] = copy.deepcopy(received_model)
                        logging.info(
                            f"Received model {received_model[0:5]} {self.client_list[clidx]['model'][0:5]} from Client {client_id}, {clidx, client}")
            # TODO:
            # Make all the clients return their unique id with all the return
            # results as the future is in undeterministic order

            self.federated_averaging()
            self.sendModel(firstInitFlag=False)
            # self.sendXtheta()
            self.calculateTrainingLoss()
        logging.info(f"***********\n**********\n Training Loss: {self.loss}")

    def sendModel(self, firstInitFlag=False):
        executor = concurrent.futures.ThreadPoolExecutor(len(self.client_list))
        futures = {executor.submit(sendModel, [client.get("client_id"),
                                               client.get("channel"),
                                               firstInitFlag, self.global_model]): client.get("client_id")
                   for client in self.client_list}
        results, _ = concurrent.futures.wait(futures)
        logging.info(
            f"Server/Hub {self.config.get('index')} finished sending data to clients: {self.client_connections.keys()}.")

    def clientSelection(self, clients_per_round):
        sampled_clients = super(MLServer, self).clientSelection(clients_per_round)
        return sampled_clients

    def queryModelFromClients(self):
        super(MLServer, self).queryModelFromClients()

    def aggregateClientModels(self):
        """
        Find the aggregate of updates from the clients selected in the current round
        """
        self.federated_averaging()

    def federated_averaging(self):
        num_clients = len(self.client_list)
        theta_sum = np.zeros((self.X_train.shape[1], 1))
        for client in self.client_list:
            theta_sum += client.get("model")
        self.global_model = copy.deepcopy(theta_sum) / num_clients
        logging.info(msg=f"Averaged global model: {self.global_model}")


