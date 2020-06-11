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
from io import BytesIO
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
def initializeDataClient(i, stub_list):
    empty = functions_pb2.Empty(value=1)
    res = stub_list[i].GenerateData(empty)
    print(f"Node {i} initialized data: {res}")


@args_wrapper
def initializeClient(i, stub_list):
    print("------------------", i, stub_list)
    l = np.random.randint(0, 256, size=(60000, 784))
    buff = BytesIO()
    np.save(buff, l, allow_pickle=False)
    initialString = functions_pb2.InitialParams(index=float(i), dc_index=int(i), device_index=int(i),
                                                stuff=buff.getvalue())
    # print(f"Sending {initialString} to client {i}")
    res = stub_list.get(i).InitializeParams(initialString)
    print(f"Node {i} initialized as client {i}, result: {res}")


@args_wrapper
def sendModel(i, opt, stub_list):
    if opt == 2:
        filename = "InitModel.h5"
    else:
        filename = "optimised_model.h5"
    ModelString = functions_pb2.Model(model=encode_file(filename))
    res = stub_list[i].SendModel(ModelString)
    print("node0", i + 1, ":", res.value, " - file :", filename)


@args_wrapper
def trainFunc(client_id, stub_list):
    empty = functions_pb2.Empty(value=1)
    res = stub_list[client_id].Train(empty)
    with open("Models/model_" + str(i + 1) + ".h5", "wb") as file:
        file.write(base64.b64decode(res.model))
    print("Saved model from node0", client_id)


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
