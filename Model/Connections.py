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
        self._node_id = node
        self._weight = None

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value


class Connections:
    """
    Connections in a neuron
    """
    def __init__(self):
        self._connections = np.array([])

    @property
    def connections(self):
        return self._connections

    def add_connection(self, connection):
        self._connections = np.append(self._connections, connection)

