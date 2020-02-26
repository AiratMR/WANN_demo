import numpy as np
from scipy.special import expit


def linear(x):
    return x


def inverse(x):
    return -x


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return expit(x)


def relu(x):
    return np.maximum(0, x)


def step(x):
    return 1 if x >= 0 else 0


def sine(x):
    return np.sin(x)


def cosine(x):
    return np.cos(x)


def gaussian(x):
    return np.exp(-(x * x) / 2)


def absolute(x):
    return np.abs(x)


FUNCTIONS = {
    'linear': linear,
    'inverse': inverse,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'relu': relu,
    'step': step,
    'sine': sine,
    'cosine': cosine,
    'gaussian': gaussian,
    'absolute': absolute
}


