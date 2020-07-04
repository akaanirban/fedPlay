#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/16/20 3:39 PM 2020

@author: Anirban Das
"""
import numpy as np


def linear_cost(theta, x, y, lambduh=0):
    """
    Regularized linear loss for l2 regression or Ridge Regression
    :param theta: The weight vector
    :param x: The data matrix
    :param y: The target vector
    :param lambduh: The regularization parameter
    :return:
    """
    m = len(x)
    residual = x @ theta - y
    # this is ridge regression
    cost_ = 1 / (2 * m) * (residual.T @ residual) + lambduh / 2 * theta.T @ theta
    return cost_[0, 0]