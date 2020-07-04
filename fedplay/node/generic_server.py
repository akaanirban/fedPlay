#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 01:20:30 2020

@author: Anirban Das
"""

from abc import ABC, abstractmethod
import logging
import grpc
import concurrent.futures
import copy
from fedplay.node.client_rpc_utils_for_server import initializeClient, getClientModels
from fedplay.node.proto import functions_pb2, functions_pb2_grpc
import random
import numpy as np
from datetime import datetime


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

    def initializeNetwork(self):
        logging.info(f"Server/Hub {self.config.get('index')} trying to find clients.")
        for idx, client in enumerate(self.client_config):
            client_address = self.client_config.get(client)
            # Channel and the port for the client
            # Modify the default buffer size : https://github.com/tensorflow/serving/issues/1382#issuecomment-503375968
            channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                           ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
            client_channel = grpc.insecure_channel(f'{client_address}', options=channel_opt)
            # Client Stub
            client_stub = functions_pb2_grpc.FederatedAppStub(client_channel)
            self.client_connections[int(client)] = client_stub  # [client_channel, client_stub]
            # threading.Thread(target=listen_for_messages, args=(channel05, stub05,), daemon=True).start()
            self.client_list.append({"address": client_address,
                                     "channel": client_stub,
                                     "model": None,
                                     "client_id": "",
                                     "client_sequence": idx})
        # Initialize a client in each separate threads
        executor = concurrent.futures.ThreadPoolExecutor(len(self.client_list))
        fut = {executor.submit(initializeClient, [client.get("client_sequence"),
                                                  client.get("channel"),
                                                  np.zeros((self.coordinates_per_dc, 1))]): client.get(
            "client_sequence") for
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

    def clientSelection(self, clients_per_round=-1):
        logging.info(msg=f"Selected {len(self.sampled_clients)}")
        sampled_clients = self.client_list
        if clients_per_round > 0:
            sampled_clients = random.sample(self.client_list, clients_per_round)
        self.sampled_clients = sampled_clients
        logging.info(msg=f"Selected {len(self.sampled_clients)}")
        return sampled_clients

    def queryModelFromClients(self):
        executor = concurrent.futures.ThreadPoolExecutor(len(self.client_list))
        futures = {executor.submit(getClientModels,
                                   [client.get("client_id"),
                                    client.get("channel")]): client.get("client_id") for client in self.client_list}
        results, _ = concurrent.futures.wait(futures)
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

    def saveModelLoss(self):
        """
        Saved the model loss in the current directory
        :return: None
        """
        loss = np.array(self.loss)
        timestamp = str(datetime.now().timestamp()).split('.')[0]
        np.save(f"results_{timestamp}.npy", loss)
        np.save(f"model_{timestamp}.npy", self.global_model)
        logging.info(msg=f"Saved current model and losses at results_{timestamp}.npy and model_{timestamp}.npy")