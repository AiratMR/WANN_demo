"""
The module contains a definitions of a basic neural network element
"""
from abc import abstractmethod

from .Connections import Connections


class BaseNode:
    """
    Base node of neural network
    """
    def __init__(self, node_id,
                 layer_id):
        self.node_id = node_id
        self.layer_id = layer_id
        self.value = None

    @abstractmethod
    def evaluate_node_output(self):
        pass


class InputNode(BaseNode):
    """
    Node in input layer of neural network
    """
    def __init__(self, node_id,
                 layer_id,
                 input_value=None):
        super().__init__(node_id, layer_id)
        self.input_value = input_value

    def evaluate_node_output(self):
        self.value = self.input_value


class HiddenNode(BaseNode):
    """
    Node in hidden layer of neural network
    """

    def __init__(self, node_id,
                 layer_id,
                 activation):
        super().__init__(node_id, layer_id)
        self.activation = activation
        self.prev_connections = Connections()

    def evaluate_node_output(self):
        aggregation_value = 0
        for connection in self.prev_connections.connections:
            aggregation_value += connection.weight * connection.node.value

        self.value = self.activation(aggregation_value)


class OutputNode(BaseNode):
    """
    Node in hidden layer of neural network
    """

    def __init__(self, node_id,
                 layer_id,
                 activation):
        super().__init__(node_id, layer_id)
        self.activation = activation
        self.prev_connections = Connections()

    def evaluate_node_output(self):
        aggregation_value = 0
        for connection in self.prev_connections.connections:
            aggregation_value += connection.weight * connection.node.value

        self.value = self.activation(aggregation_value)
