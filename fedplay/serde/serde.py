#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/15/20 8:38 PM 2020

@author: Anirban Das
"""

import numpy as np
from io import BytesIO


def serialize(array: np.ndarray) -> bytes:
    """
    Serializes a Numpy NDArray into a byte string
    :param array: The ndarray to be serialized
    :return: a bytes string
    """
    buff = BytesIO()
    np.save(buff, array, allow_pickle=False)
    return buff.getvalue()


def deserialize(buff: str) -> np.ndarray:
    """
    Deserializes a Numpy array received in bytes format
    :param buff: The string/bytes to be deserialized into Numpy array
    :return: Deserialized numpy array
    """
    temp = BytesIO(buff)
    arr = np.load(temp, allow_pickle=False)
    return arr
