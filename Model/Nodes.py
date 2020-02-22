"""
The module contains a definitions of a basic neural network element
"""
from abc import abstractmethod

from .Connections import Connections


class BaseNode:
    """
    Base node of neural network
    """
    def __init__(self, node_id: int,
                 layer_id: int,
                 weight: float,
                 activation):
        self._node_id = node_id
        self._layer_id = layer_id
        self._weight = weight
        self._activation = activation
        self._eval_result = None

    @abstractmethod
    def evaluate_node_output(self):
        pass


class InputNode(BaseNode):
    """
    Input node of neural network
    """
    def __init__(self, node_id: int,
                 layer_id: int,
                 weight: float,
                 activation,
                 input_value):
        super().__init__(node_id, layer_id, weight, activation)
        self._input_value = input_value

    def evaluate_node_output(self):
        self._eval_result = self._activation(self._input_value)


class HiddenNode(BaseNode):
    """
    Node of neural network
    """

    def __init__(self, node_id: int,
                 layer_id: int,
                 weight: float,
                 activation):
        super().__init__(node_id, layer_id, weight, activation)
        self._connections = Connections()

    def evaluate_node_output(self):
        input_values = self._connections.

