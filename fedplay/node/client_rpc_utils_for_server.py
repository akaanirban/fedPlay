#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/10/20 12:25 PM 2020

@author: Anirban Das
"""

from proto import functions_pb2 as functions_pb2
import base64
import functools
import logging
from node_utils import serialize, deserialize
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO)


def args_wrapper(func):
    # https://realpython.com/primer-on-python-decorators/
    # Allows the functions here to accept multiple arguments when
    # called from executor.submit
    @functools.wraps(func)
    def wrapper_decorator(args):
        try:
            value = func(*args)
        except Exception as e:
            logging.info(f"Exception: {e}")
        return value

    return wrapper_decorator


@args_wrapper
def encode_file(file_path):
    with open(file_path, 'rb') as file:
        encoded_string = base64.b64encode(file.read())
    return encoded_string


@args_wrapper
def encode_model(model):
    encoded_string = base64.b64encode(model)
    return encoded_string


@args_wrapper
def initializeDataClient(client_id, stub, X, y):
    # logging.debug(f"To send {client_id} got {X.shape}, {y.shape}, \n {X[0:5, 0:5]}")
    data = functions_pb2.Data(x=serialize(X), y=serialize(y))
    res = stub.InitializeData(data)
    logging.info(msg=f"Node {client_id} initialized with #datapoints: {res.numeric_reply} from client {res.str_reply}")
    return client_id


@args_wrapper
def initializeClient(i, stub, model):
    initialString = functions_pb2.InitialParams(index=float(i), dc_index=int(i), device_index=int(i),
                                                model=serialize(model), alpha=0.01, lambduh=0.01)
    # print(f"Sending {initialString} to client {i}")
    res = stub.InitializeParams(initialString)
    logging.debug(f"Node {i} initialized as client {i}, result: {res}")
    return i, res


@args_wrapper
def sendModel(client_id, stub, firstInitFlag, global_model):
    model = functions_pb2.Model(model=serialize(global_model))
    res = stub.UpdateLocalModels(model)
    xtheta = deserialize(res.xtheta)
    logging.info(msg=f"Client {res.id} sends back Xtheta {xtheta[0:10]} ")
    assert res.id == client_id
    return client_id, xtheta


@args_wrapper
def trainFunc(client_id, stub, q, lambduh, xtheta, model=None):
    trainconfig = functions_pb2.TrainConfig()
    trainconfig.q = q
    trainconfig.lambduh = lambduh
    trainconfig.model.xtheta = serialize(xtheta)
    #trainconfig.model.model = serialize(model)

    # if xtheta:
    #     trainconfig.model.xtheta = serialize(xtheta)
    # if model:
    #     trainconfig.model.model = serialize(model)
    # if not xtheta and not model:
    #     raise Exception
    res = stub.Train(trainconfig)
    model = deserialize(res.model)
    logging.info(f"Server received model from Client {client_id} ==> {model.shape}")
    return client_id, model

@args_wrapper
def getClientModel(client_id, stub_list):
    """
    get the current model from a client
    """
    empty = functions_pb2.Empty(value=1)
    print(f"Getting model from client {client_id}")
    res = stub_list[client_id].SendModelToServer(empty)
    print(f"Node {client_id}: Obtained model from client {client_id}, result: {res}")
    return res
