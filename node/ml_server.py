#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 01:28:12 2020

@author: Anirban Das
"""

import grpc
import concurrent.futures
import collections
from node_utils import *
from client_rpc_utils_for_server import initializeClient, encode_file
from generic_server import Node
from proto import functions_pb2 as functions_pb2, functions_pb2_grpc as functions_pb2_grpc
import logging
logging.basicConfig(level=logging.INFO)


class MLServer(Node):
    """
        A federated learning server | Also acts as a hub
        This server is for ML based approach. Not DL
    """

    def __init__(self, config):
        logging.info(f"Booting Server/Hub {config.get('index')}")
        self.config = config
        self.client_config = config.get('client_config')
        # self.initialize()
        self.client_connections = collections.OrderedDict()
        self.sampled_clients = collections.OrderedDict()

    def initialize(self):
        self.alpha: float = self.config.get('alpha')
        self.Xtheta: np.array = self.config.get('Xtheta')
        self.costs = []
        self.theta = self.config.get('theta')
        self.X = self.config.get('X')
        self.y = self.config.get('y')
        self.index = self.config.get('index')
        self.offset = self.config.get('offset')
        self.theta_average = self.theta[self.index * self.offset: (self.index + 1) * self.offset]
        self.local_estimate = None
        # From https://github.com/akaanirban/BlockCoordinateDescent/blob/c18622aeb8f08cba87d9ad47e4b1b252e93b213a/loss_functions.py#L78
        self.lipschitz_constant = np.max(np.linalg.eig(self.X.T @ self.X)[0])
        self.global_Xtheta = None

    def initializeNetwork(self):
        logging.info(f"Server/Hub {self.config.get('index')} trying to find clients.")
        for client in self.client_config:
            client_port = self.client_config.get(client)
            # Channel and the port for the client
            # Modify the default buffer size : https://github.com/tensorflow/serving/issues/1382#issuecomment-503375968
            channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
            client_channel = grpc.insecure_channel(f'localhost:{client_port}', options=channel_opt)
            # Client Stub
            client_stub = functions_pb2_grpc.FederatedAppStub(client_channel)
            self.client_connections[int(client)] = client_stub#[client_channel, client_stub]
            # threading.Thread(target=listen_for_messages, args=(channel05, stub05,), daemon=True).start()
        # Initialize a client in each separate threads
        n_clients = len(self.client_connections)
        executor = concurrent.futures.ThreadPoolExecutor(n_clients)
        fut = [executor.submit(initializeClient, [i, self.client_connections]) for i in range(n_clients)]
        print(fut)
        concurrent.futures.wait(fut)
        logging.info(
            f"Server/Hub {self.config.get('index')} finished setup of clients: {self.client_connections.keys()}.")

    def initializeData(self):
        n_clients = len(self.client_connections)
        executor = concurrent.futures.ThreadPoolExecutor(n_clients)
        futures = [executor.submit(initializeDataClient, i) for i in range(n_clients)]
        concurrent.futures.wait(futures)
        logging.info(
            f"Server/Hub {self.config.get('index')} finished data setup of clients: {self.client_connections.keys()}.")

    def sendModel(self, firstInitFlag):
        n_clients = len(self.client_connections)
        executor = concurrent.futures.ThreadPoolExecutor(n_clients)
        futures = [executor.submit(sendModel, [i, firstInitFlag]) for i in range(n_clients)]
        concurrent.futures.wait(futures)
        logging.info(
            f"Server/Hub {self.config.get('index')} finished sending data to clients: {self.client_connections.keys()}.")

    def train(self, clients_per_round=0):
        for global_round in range(self.config.get('T')):
            logging.info(
                f"Server/Hub {self.config.get('index')} starting global round {global_round}.")
            sampled_clients = self.clientSelection(clients_per_round)
            executor = concurrent.futures.ThreadPoolExecutor(len(sampled_clients))
            futures = [executor.submit(trainFunc, client_id) for client_id in range(len(sampled_clients))]
            concurrent.futures.wait(futures)

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

    def clientSelection(self, clients_per_round):
        sampled_clients = self.client_connections
        if clients_per_round > 0:
            temp = [client for client in random.sample(
                list(self.client_connections), clients_per_round)]
            sampled_clients = {j: self.client_connections.get(j) for j in temp}
        self.sampled_clients = sampled_clients
        return sampled_clients

    def optimiseModels(self):
        pass
        # models = [load_model("Models/model_" + str(i) + ".h5") for i in range(1, 6)]
        # weights = [model.get_weights() for model in models]
        #
        # new_weights = list()
        #
        # for weights_list_tuple in zip(*weights):
        #     new_weights.append(
        #         [np.array(weights_).mean(axis=0) \
        #          for weights_ in zip(*weights_list_tuple)])
        #
        # new_model = models[0]
        # new_model.set_weights(new_weights)
        # new_model.save("Models/optimised_model.h5")
        # print("Averaged over all models - optimised model saved!")

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


if __name__ == "__main__":
    server = MLServer({"index": 1, "client_config": {"1": 8081, "2": 8082, "3": 8083, "4": 8084, "5": 8085}})
    while True:
        print("1. Generate Data")
        print("2. Initialize model on all nodes")
        print("3. Perform training on all nodes")
        print("4. Average and optimize new model")
        print("5. Send new model to all nodes")
        print("6. initializeNetwork")
        print("Enter an option: ")
        option = input()

        if (option == "1"):
            generateData()
        if (option == "2"):
            sendModel(int(option))
        if (option == "3"):
            train()
        if (option == "4"):
            optimiseModels()
        if (option == "5"):
            sendModel(int(option))
        if option == "6":
            print(":sdsds")
            server.initializeNetwork()
