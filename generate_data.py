import numpy as np
import torch
import random
import math
from collections import namedtuple
import os
import time

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 22
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.signal import medfilt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device Status:", device)


def get_graph_mat(n=20, size=1):
    coords = size * np.random.uniform(size=(n,2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat


def plot_graph(coords, mat):
    n = len(coords)
    annotations = ['0','1','2','3','4','5','6','7','8','9']
    plt.scatter(coords[:,0], coords[:,1], s=[50 for _ in range(n)])

    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'b', alpha=0.7)
    # plt.show()
    k = 0
    for i, j in zip(coords[:, 0], coords[:, 1]):
        #         print("i here", i)
        plt.text(i, j, '({})'.format(k))
        k = k + 1


# coords, W_np = get_graph_mat(n=10)
# plot_graph(coords, W_np)
