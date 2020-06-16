#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/15/20 9:24 PM 2020

@author: Anirban Das
"""
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)
import fedplay as play
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("./superconductivity/train.csv")
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
coordinates_per_dc = int(X_train_minmax.shape[1] / N)
extradatapointsinfirstdevice = X_train_minmax.shape[1] - coordinates_per_dc * N
# largst_SV = np.max(np.linalg.eig(X_train_minmax[:, 1:].T @ X_train_minmax[:, 1:])[0])
datapoints_per_device = int(X_train_minmax.shape[0] / (K))
# alpha = 0.00000001 # for the case when X is not notmalized
alpha = 0.01  # 0.031 # = 16000/505347
lambduh = 0.01
decreasing_step = False
decreasing_step = False


if __name__ == "__main__":
    # server = MLServer(
    #     {"index": 1, "T": 100, "lambduh": 0.01,
    #      "client_config": {"1": 8081, "2": 8082, "3": 8083, "4": 8084, "5": 8085}})
    server = play.MLServer({"index": 1,
                            "T":20,
                            "lambduh": 0.01,
                            "X_train": X_train_minmax,
                            "y_train": binary_y,
                            "coordinates_per_dc": coordinates_per_dc,
                            "datapoints_per_device": datapoints_per_device,
                            "client_config": {"1": "localhost:5081", "2": "localhost:5082", "3": "localhost:5083", "4": "localhost:5084", "5": "localhost:5085"}
                            # "client_config": {"1": 8080}
                            })

    #server.initializeNetwork()
    #print(f"Clients are {server.client_list}")
    #server.initializeData()
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




