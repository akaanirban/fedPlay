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
from proto import functions_pb2 as functions_pb2, functions_pb2_grpc as functions_pb2_grpc
import client_functions as functions

_ONE_HR_IN_SECONDS = 3600


class Client(functions_pb2_grpc.FederatedAppServicer):

    def __init__(self, config=None) -> None:
        self.alpha: float = None #alpha
        self.theta = None#theta
        self.X = None#X
        self.y = None#y
        self.device_index = None#device_index
        self.dc_index = None#dc_index
        self.offset = None#offset
        self.Xtheta = None#Xtheta  # Xtheta is the predicted y using data from all coordinates
        # self.alpha = len(self.X)/max(np.linalg.eig(self.X.T @ self.X)[0])

    def InitializeParams(self, request, context):
        response = functions_pb2.Empty()
        print(f"Node: {request.index} received an initial string of type {type(request)} and value: {request}")
        self.alpha = request.alpha
        self.device_index = request.device_index
        self.dc_index = request.dc_index
        logging.info(msg=f"Client {self.device_index} is initialized")
        return response

    def GenerateData(self, request, context):
        response = functions_pb2.Empty()
        response.value = functions.GenerateData()
        return response

    def SendModel(self, request, context):
        response = functions_pb2.Empty()
        response.value = functions.SendModel(request.model)
        return response

    def Train(self, request, context):
        response = functions_pb2.Model()
        response.model = functions.Train()
        return response

    def SendModelToServer(self, request, context):
        response = functions_pb2.Model()
        response.model = functions.Train()
        return response


if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    functions_pb2_grpc.add_FederatedAppServicer_to_server(Client(), server)

    print("Starting server on PORT: 8080")
    server.add_insecure_port('[::]:8080')
    server.start()

    try:
        while True:
            time.sleep(_ONE_HR_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
