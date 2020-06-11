#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:03:22 2020

@author: Anirban Das

"""

import numpy as np
import random
import pickle
import os
import logging


def listen_for_messages(client, conn):
    pass
    """
    This method will be ran in a separate thread as the main/ui thread, because the for-in call is blocking
    when waiting for new messages
    """
    # for note in conn.ChatStream(chat.Empty()):  # this line will wait for new messages from the server!
    #     print("R[{}] {}".format(note.name, note.message))  # debugging statement


def federatedAveraging():
    pass
    # import
    # fl_model  # pylint: disable=import-error
    #
    # # Extract updates from reports
    # updates = self.extract_client_updates(reports)
    #
    # # Extract total number of samples
    # total_samples = sum([report.num_samples for report in reports])
    #
    # # Perform weighted averaging
    # avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
    #               for _, x in updates[0]]
    # for i, update in enumerate(updates):
    #     num_samples = reports[i].num_samples
    #     for j, (_, delta) in enumerate(update):
    #         # Use weighted average by number of samples
    #         avg_update[j] += delta * (num_samples / total_samples)
    #
    # # Extract baseline model weights
    # baseline_weights = fl_model.extract_weights(self.model)
    #
    # # Load updated weights into model
    # updated_weights = []
    # for i, (name, weight) in enumerate(baseline_weights):
    #     updated_weights.append((name, weight + avg_update[i]))
    #
    # return updated_weights


# Server operations
def flatten_weights(weights):
    # Flatten weights into vectors
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())

    return np.array(weight_vecs)


def set_client_data(self, client):
    loader = self.config.loader

    # Get data partition size
    if loader != 'shard':
        if self.config.data.partition.get('size'):
            partition_size = self.config.data.partition.get('size')
        elif self.config.data.partition.get('range'):
            start, stop = self.config.data.partition.get('range')
            partition_size = random.randint(start, stop)

    # Extract data partition for client
    if loader == 'basic':
        data = self.loader.get_partition(partition_size)
    elif loader == 'bias':
        data = self.loader.get_partition(partition_size, client.pref)
    elif loader == 'shard':
        data = self.loader.get_partition()
    else:
        logging.critical('Unknown data loader type')

    # Send data to client
    client.set_data(data, self.config)


def save_model(self, model, path, identifier, ml_dl_flag="ML"):
    path += '/global'
    if ml_dl_flag == "DL":
        import torch
        torch.save(model.state_dict(), path)
    else:
        # save the classifier
        with open(os.path.join(path, 'classifier_{identifier}.pkl'), 'wb') as fid:
            pickle.dump(model, fid)

    logging.info('Saved global model: {}'.format(path))


def save_reports(self, round, reports):
    pass
    # Need to remodel this
    # if reports:
    #     self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
    #         report.weights)) for report in reports]
    #
    # # Extract global weights
    # self.saved_reports['w{}'.format(round)] = self.flatten_weights(
    #     fl_model.extract_weights(self.model))
