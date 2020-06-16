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
from node_utils import *
from client_rpc_utils_for_server import initializeClient, encode_file, initializeDataClient, trainFunc, sendModel
from generic_server import Node
from proto import functions_pb2 as functions_pb2, functions_pb2_grpc as functions_pb2_grpc
import logging
import sys
import copy
from datetime import datetime
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import numpy as np

logging.basicConfig(level=logging.WARNING)

df = pd.read_csv("/home/anirban/Softwares/GitHub/fed_hierarchy/data/superconductivity/train.csv")
df = shuffle(df, random_state=42)
X = df.iloc[:16000, 0: -1]  # 16000 will be 75% of the data
y = df.iloc[:16000, -1]
mm_scaler = StandardScaler()
X_train_minmax = mm_scaler.fit_transform(X)
X_train_minmax = np.array(X_train_minmax)
X_train_minmax = np.insert(X_train_minmax, 0, 1, axis=1)
binary_y = np.array(y).reshape(-1, 1)
X_test = df.iloc[16000:, :-1]
X_test = mm_scaler.transform(X_test)
# X_test = np.array(X_test)
X_test = np.insert(X_test, 0, 1, axis=1)
y_test = df.iloc[16000:, -1]

del X
del y
del df

N = 1
K = 5
global_epoch = 10
local_epoch = 10
coordinate_per_dc = int(X_train_minmax.shape[1] / N)
extradatapointsinfirstdevice = X_train_minmax.shape[1] - coordinate_per_dc * N
# largst_SV = np.max(np.linalg.eig(X_train_minmax[:, 1:].T @ X_train_minmax[:, 1:])[0])
datapoints_per_device = int(X_train_minmax.shape[0] / (K))
# alpha = 0.00000001 # for the case when X is not notmalized
alpha = 0.01  # 0.031 # = 16000/505347
lambduh = 0.01
decreasing_step = False
decreasing_step = False


class MLServer(Node):
    """
        A federated learning server | Also acts as a hub
        This server is for ML based approach. Not DL
    """

    def __init__(self, config):
        logging.info(f"Booting Server/Hub {config.get('index')}")
        self.config = config
        self.client_config = config.get('client_config')
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
        self.global_Xtheta: np.array = np.zeros((X_train_minmax.shape[0], 1))  # self.config.get('Xtheta')
        self.global_model: np.array = np.zeros((X_train_minmax.shape[1], 1))  # self.config.get('theta')
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
        logging.info(f"Server/Hub {self.config.get('index')} trying to find clients.")
        for idx, client in enumerate(self.client_config):
            client_port = self.client_config.get(client)
            # Channel and the port for the client
            # Modify the default buffer size : https://github.com/tensorflow/serving/issues/1382#issuecomment-503375968
            channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                           ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
            client_channel = grpc.insecure_channel(f'localhost:{client_port}', options=channel_opt)
            # Client Stub
            client_stub = functions_pb2_grpc.FederatedAppStub(client_channel)
            self.client_connections[int(client)] = client_stub  # [client_channel, client_stub]
            # threading.Thread(target=listen_for_messages, args=(channel05, stub05,), daemon=True).start()
            self.client_list.append({"port": client_port,
                                     "channel": client_stub,
                                     "model": None,
                                     "client_id": "",
                                     "client_sequence": idx})
        # Initialize a client in each separate threads
        executor = concurrent.futures.ThreadPoolExecutor(len(self.client_list))
        fut = {executor.submit(initializeClient, [client.get("client_sequence"),
                                                  client.get("channel"),
                                                  np.zeros((coordinate_per_dc, 1))]): client.get("client_sequence") for
               idx, client in enumerate(self.client_list)}
        results, _ = concurrent.futures.wait(fut)
        for r in results:
            # Parse the client id received and the corresponding sequence id
            res = r.result()
            client_sequence = res[0]
            received_client_id = res[1].str_reply
            # Set the received client ID from the client in the data structure
            for idx, client in enumerate(self.client_list):
                if client.get("client_sequence") == client_sequence:
                    self.client_list[idx]["client_id"] = received_client_id
        logging.info(f"\nThe initialized client list:\n {self.client_list} \n\n")
        logging.info(
            f"Server/Hub {self.config.get('index')} finished setup of clients: {[client.get('client_id') for client in self.client_list]}.")

    def initializeData(self):
        executor = concurrent.futures.ThreadPoolExecutor(len(self.client_list))
        futures = {executor.submit(initializeDataClient,
                                   [client.get("client_id"), client.get("channel"),
                                    X_train_minmax[client.get("client_sequence") * datapoints_per_device: (client.get(
                                        "client_sequence") + 1) * datapoints_per_device, :],
                                    binary_y[client.get("client_sequence") * datapoints_per_device: (client.get(
                                        "client_sequence") + 1) * datapoints_per_device, :]]
                                   ): client.get("client_id") for client in self.client_list}
        results, _ = concurrent.futures.wait(futures)
        logging.info(
            f"Server/Hub {self.config.get('index')} finished data setup of clients: {[res.result() for res in results]}.")

    def trainingLoss(self):
        self.loss.append(self.linear_cost(self.global_model, X_train_minmax, binary_y, self.lambduh))

    @staticmethod
    def linear_cost(theta, x, y, lambduh=0):
        m = len(x)
        residual = x @ theta - y
        # this is ridge regression
        cost_ = 1 / (2 * m) * (residual.T @ residual) + lambduh / 2 * theta.T @ theta
        return cost_[0, 0]

    def train(self, clients_per_round=0):
        for global_round in range(self.config.get('T')):
            logging.info(
                f"Server/Hub {self.config.get('index')} starting global round {global_round}.")
            sampled_clients = self.clientSelection(clients_per_round)
            executor = concurrent.futures.ThreadPoolExecutor(len(sampled_clients))
            futures = {executor.submit(trainFunc,
                                       [client.get("client_id"), client.get("channel"),
                                        self.local_epoch, self.lambduh,
                                        self.global_Xtheta[client.get("client_sequence") * datapoints_per_device:
                                                    (client.get("client_sequence") + 1) * datapoints_per_device, :], "",
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
                        print(
                            f"Received model {received_model[0:5]} {self.client_list[clidx]['model'][0:5]} from Client {client_id}, {clidx, client}")
            # TODO:
            # Make all the clients return their unique id with all the return
            # results as the future is in undeterministic order

            self.federated_averaging()
            self.sendModel(firstInitFlag=False)
            # self.sendXtheta()
            self.trainingLoss()
        logging.info(f"***********\n**********\n Training Loss: {self.loss}")

    def sendModel(self, firstInitFlag):
        executor = concurrent.futures.ThreadPoolExecutor(len(self.client_list))
        futures = {executor.submit(sendModel, [client.get("client_id"),
                                               client.get("channel"),
                                               firstInitFlag, self.global_model]): client.get("client_id")
                   for client in self.client_list}
        results, _ = concurrent.futures.wait(futures)
        self.global_Xtheta = np.zeros((X_train_minmax.shape[0], 1))
        # TODO: Fix this such that the stacking is based on client id
        for idx, temp in enumerate(results):
            print("Inside send model", temp)
            client_id, client_xtheta = temp.result()
            for clidx, client in enumerate(self.client_list):
                if client.get("client_id") == client_id:
                    sequence = self.client_list[clidx]["client_sequence"]
            self.global_Xtheta[sequence * datapoints_per_device:(sequence + 1) * datapoints_per_device,
            :] = copy.deepcopy(client_xtheta)
        logging.info(
            f"Server/Hub {self.config.get('index')} finished sending data to clients: {self.client_connections.keys()} and {self.global_Xtheta[0:10]}.")

    def clientSelection(self, clients_per_round):
        sampled_clients = self.client_list
        if clients_per_round > 0:
            sampled_clients = random.sample(self.client_list, clients_per_round)
        self.sampled_clients = sampled_clients
        logging.debug(msg=f"Selected {len(self.sampled_clients)}")
        return sampled_clients

    def federated_averaging(self):
        num_clients = len(self.client_list)
        #print(self.client_list[0]["model"])
        theta_sum = np.zeros((X_train_minmax.shape[1], 1))
        for client in self.client_list:
            theta_sum += client.get("model")
        self.global_model = copy.deepcopy(theta_sum) / num_clients
        # logging.info(msg=f"Averaged global model: {self.global_model}")

    def queryModelFromClients(self):
        n_clients = len(self.client_connections)
        executor = concurrent.futures.ThreadPoolExecutor(n_clients)
        futures = [executor.submit(getClientModel, [i, self.client_connections]) for i in range(n_clients)]
        concurrent.futures.wait(futures)
        clientModels = []
        for future in futures:
            # https://stackoverflow.com/a/52082992/8853476
            clientModels.append(future.result())
        logging.info(
            f"Server/Hub {self.config.get('index')} obtained models from clients: {clientModels}.")

    def saveModelLoss(self):
        loss = np.array(self.loss)
        timestamp = str(datetime.now().timestamp()).split('.')[0]
        np.save(f"results_{timestamp}.npy", loss)
        np.save(f"model_{timestamp}.npy", self.global_model)
        logging.info(msg=f"Saved current model and losses at results_{timestamp}.npy and model_{timestamp}.npy")

    def aggregateClientModels(self):
        """
        Find the updates from the clients selected in the current round
        """
        n_clients = len(self.sampled_clients)
        executor = concurrent.futures.ThreadPoolExecutor(n_clients)
        futures = [executor.submit(getClientModel, i) for i in range(n_clients)]
        futures, _ = concurrent.futures.wait(futures)
        clientModels = collections.OrderedDict()
        for idx, future in enumerate(futures):
            # https://stackoverflow.com/a/52082992/8853476
            clientModels[idx] = future.result()
        # federatedAveraging(result)
        logging.info(
            f"Server/Hub {self.config.get('index')} finished sending data to clients: {self.client_connections.keys()}.")


if __name__ == "__main__":
    server = MLServer(
        {"index": 1, "T": 100, "lambduh": 0.01,
         "client_config": {"1": 8081, "2": 8082, "3": 8083, "4": 8084, "5": 8085}})
    while True:
        print("1. Initialize Network ")
        print("2. Initialize Data on all clients")
        print("3. Start Training on clients")
        print("4. Save Model and global Loss")
        print("5. Test on the global model")
        print("6. Exit")
        print("Enter an option: ")
        option = input()

        if option == "1":
            server.initializeNetwork()
        if option == "2":
            server.initializeData()
        if option == "3":
            server.train(clients_per_round=-1)
        if option == "4":
            server.saveModelLoss()
        if option == "5":
            sendModel(int(option))
        if option == "6":
            sys.exit(0)
