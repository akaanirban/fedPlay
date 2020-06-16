#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/10/20 1:37 PM 2020

@author: Anirban Das
"""

import grpc
from concurrent import futures
import time
import logging
import numpy as np
from io import BytesIO
from node_utils import serialize, deserialize
from proto import functions_pb2 as functions_pb2, functions_pb2_grpc as functions_pb2_grpc
import client_functions as functions
import uuid
logging.basicConfig(level=logging.INFO)
_ONE_HR_IN_SECONDS = 3600


class Client(functions_pb2_grpc.FederatedAppServicer):

    def __init__(self, config=None) -> None:
        self.alpha: float = None  # alpha
        self.theta = None  # theta
        self.X = None  # X
        self.y = None  # y
        self.device_index = None  # device_index
        self.dc_index = None  # dc_index
        self.offset = None  # offset
        self.Xtheta = None  # Xtheta  # Xtheta is the predicted y using data from all coordinates
        self.global_rounds_counter = 0
        # self.alpha = len(self.X)/max(np.linalg.eig(self.X.T @ self.X)[0])
        self.identifier = str(uuid.uuid4())

    def InitializeParams(self, request, context):
        response = functions_pb2.Reply(str_reply=self.identifier)
        # print(f"Node: {request.index} received an initial string of type {type(request)} and value: {request}")
        self.alpha = request.alpha
        self.lambduh = request.lambduh
        self.device_index = request.device_index
        self.dc_index = request.dc_index
        self.theta = deserialize(request.model)
        self.decreasing_step = request.decreasing_step
        logging.info(msg=f"Client {self.device_index} Shape of the received array is the following: {self.theta.shape}")
        logging.info(msg=f"Client {self.device_index} is initialized with identifier {self.identifier}")
        return response

    def GenerateData(self, request, context):
        response = functions_pb2.Empty()
        response.value = functions.GenerateData()
        return response

    def InitializeData(self, request, context):
        self.X = deserialize(request.x)
        self.y = deserialize(request.y)
        logging.info(msg=f"Client {self.device_index} initialized X {self.X.shape} and Y {self.y.shape}")
        response = functions_pb2.Reply(str_reply=self.identifier, numeric_reply=self.X.shape[0]) # Returns the number of samples
        # logging.debug(self.identifier, "\n", self.X[0:5,0:5])
        return response

    def Train(self, request, context):
        self._increase_global_counts()
        self.Xtheta = deserialize(request.model.xtheta)
        Q = int(request.q)
        lambduh = float(request.lambduh)
        logging.info(msg=f"Client {self.device_index} the dimension of Xtheta {self.Xtheta.shape}, Theta {self.theta.shape}")
        # Isolate the H_-k from other datacenters for the same label space
        # Obtained in the last iteration
        Xtheta_from_other_DC = self.Xtheta - self.X @ self.theta  # Assuming label space is same
        for rounds in range(Q):
            print("Starting round ", rounds)
            # batch gradient descent for the time being

            # If NO partital gradient information from outside is used
            # grad = 1/len(device.X) * device.X.T @ (device.X @ device.theta - device.y)

            # If partital gradient information from outside is used
            grad = 1 / len(self.X) * self.X.T @ (
                    (Xtheta_from_other_DC + self.X @ self.theta) - self.y) + lambduh * self.theta
            if self.decreasing_step:
                self.theta = self.theta - self.alpha / np.sqrt(
                    self.global_rounds_counter + 1) * grad  # decreasing dtep size
            else:
                self.theta = self.theta - self.alpha * grad
        self.Xtheta = self.X @ self.theta  # Update the value of the predicted y (probably unnecessary and not used)
        response_model = functions_pb2.Model(model=serialize(self.theta))
        return response_model

    def SendModel(self, request, context):
        response = functions_pb2.Model()
        response.model = serialize(self.theta)
        return response

    def Test(self):
        pass

    def UpdateLocalModels(self, request, context):
        self.theta = deserialize(request.model)
        self.Xtheta = self.X @ self.theta
        response = functions_pb2.Model(xtheta=serialize(self.Xtheta), id=self.identifier)
        return response

    def _increase_global_counts(self):
        # Increases global rounds for which this device participates
        self.global_rounds_counter += 1

    def __str__(self) -> str:
        return f"<Federated Client id:{self.device_index}>"


if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_send_message_length', 512 * 1024 * 1024),
        ('grpc.max_receive_message_length', 512 * 1024 * 1024)
    ])

    functions_pb2_grpc.add_FederatedAppServicer_to_server(Client(), server)

    print("Starting server on PORT: 8080")
    server.add_insecure_port('[::]:8080')
    server.start()

    try:
        while True:
            time.sleep(_ONE_HR_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
