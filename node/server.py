#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:33:38 2020

@author: Anirban Das
"""

import os
from concurrent import futures
import time
import argparse
import grpc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import node.node_pb2 as node_pb2
import base64
import node.node_pb2_grpc as node_pb2_grpc

_ONE_HR_IN_SECONDS = 3600


class FederatedNode(node_pb2_grpc.FederatedNodeServicer):
    def __init__(self):
        self.startTime = time.time()
        df = pd.read_csv("data/superconductivity/train.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        mm_scaler = StandardScaler()
        self.X = mm_scaler.fit_transform(X)
        self.X = np.array(self.X)
        self.y = np.array(y).reshape(-1, 1)
        self.seed = 0
        np.random.seed(0)

    def InitializeData(self, request, context):
        # Set all devices to same random seed
        self.randomseed = request.randomseed

        np.random.seed(self.randomseed)
        self.alpha = request.alpha
        self.index = request.index
        self.isDC = request.isDC
        self.dc_index = request.dc_index
        self.device_index = request.device_index
        self.coordinate_per_dc = request.coordinate_per_dc
        self.datapoints_per_device = request.datapoints_per_device
        self.lambduh = request.lambduh
        # Initiate the own slice of data
        self.X = self.X[
                 self.device_index * self.datapoints_per_device: (self.device_index + 1) * self.datapoints_per_device,
                 self.dc_index * self.dc_index * self.coordinate_per_dc: (self.dc_index + 1) * self.coordinate_per_dc
                 ]
        self.y = self.y[
                 self.device_index * self.datapoints_per_device: (self.device_index + 1) * self.datapoints_per_device,
                 ]
        self.Xtheta = np.zeros((self.datapoints_per_device, 1))  # Initialize the predictions
        self.theta = np.zeros((self.X.shape[1], 1))  # Initialize the weights
        # All devices in 1st datacenter will have the intercept
        if self.dc_index == 0:
            self.X = np.insert(self.X, 0, 1, axis=1)

    def Train(self, request, context):
        local_epoch = request.local_epochs
        decreasing_step = True if request.decreasing_step_size == 1 else False
        global_epoch_idx = request.global_epoch_index
        # Isolate the H_-k from other datacenters for the same label space
        # Obtained in the last iteration
        Xtheta_from_other_DC = self.Xtheta - self.X @ self.theta  # Assuming label space is same
        for Q_idx, Q in enumerate(range()):
            # batch gradient descent for the time being

            # If NO partial gradient information from outside is used
            # grad = 1/len(device.X) * device.X.T @ (device.X @ device.theta - device.y)

            # If partial gradient information from outside is used
            grad = 1/len(self.X) * self.X.T @ ((Xtheta_from_other_DC + self.X @ self.theta) - self.y) \
                   + self.lambduh*self.theta
            if decreasing_step:
                self.theta = self.theta - self.alpha/np.sqrt(global_epoch_idx+1) * grad  # decreasing step size
            else:
                self.theta = self.theta - self.alpha * grad
        self.Xtheta = self.X @ self.theta  # Update the value of the predicted y

        self.model.save('Models/model.h5')

        with open('Models/model.h5', 'rb') as file:
            encoded_string = base64.b64encode(file.read())
        print("Model trained and saved successfully!")

    def UpdateState(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendModel(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendPredictions(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def serve(port, max_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    node_pb2_grpc.add_FederatedNodeServicer_to_server(FederatedNode(), server)
    server.add_insecure_port('[::]:{port}'.format(port=port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_HR_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


def SendModel(modelString):
    try:
        with open("Models/model.h5","wb") as file:
            file.write(base64.b64decode(modelString))
            print("Successfully saved model...")
            return 1
    except:
        print("An error occured!")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--port', type=int, help='port number', required=False, default=5052)
    parser.add_argument('--max_workers', type=int, help='# max workers', required=False, default=1)
    args = parser.parse_args()
    # Start the node server
    serve(port=args.port, max_workers=args.max_workers)
