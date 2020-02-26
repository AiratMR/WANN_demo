"""
The module contains a definition of the class of connections in a neuron
"""
import numpy as np

from Model.Nodes import BaseNode


class Connection:
    """
    Connection in a neuron
    """
    def __init__(self, node: BaseNode):
        self.node = node
        self.weight = None


class Connections:
    """
    Connections in a neuron
    """
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

