"""
The module contains a definition of the class of connections in a neuron
"""
import numpy as np


class Connection:
    """
    Connection in a neuron
    """
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight


class Connections:
    """
    Connections in a neuron
    """
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

