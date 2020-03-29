"""
The module contains the definition of the classes of layers of the neural network
"""
import random

from .Nodes import BaseNode


class BaseLayer:
    """
    Base layer
    """
    def __init__(self, layer_id, nodes=None):
        if nodes is None:
            nodes = []
        self.layer_id = layer_id
        self.nodes = nodes

    def add_node(self, node):
        """
        Add node to list of nodes
        :param node: add node -> BaseNode
        """
        self.nodes.append(node)

    def get_random_node(self):
        """
        Return random node from list of nodes
        :return: random node -> BaseNode
        """
        return random.choice(self.nodes)


class InputLayer(BaseLayer):
    """
    Input layer
    """
    def __init__(self, layer_id, nodes=None):
        super().__init__(layer_id, nodes)


class HiddenLayer(BaseLayer):
    """
    Hidden layer
    """
    def __init__(self, layer_id, nodes=None):
        super().__init__(layer_id, nodes)


class OutputLayer(BaseLayer):
    """
    Output layer
    """
    def __init__(self, layer_id, nodes=None):
        super().__init__(layer_id, nodes)

