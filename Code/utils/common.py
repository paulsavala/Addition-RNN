import numpy as np


def reverse_dict(d):
    return {v: k for k, v in d.items()}


def sigmoid(x):
    return np.exp(x)/(np.exp(x) + 1)