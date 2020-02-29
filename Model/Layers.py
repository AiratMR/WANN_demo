"""
The module contains the definition of the classes of layers of the neural network
"""
import random

from .Nodes import BaseNode


class BaseLayer:
    """
    Base layer
    """
    def __init__(self, layer_id, nodes=[]):
        self.layer_id = layer_id
        self.nodes = nodes

    def add_node(self, node: BaseNode):
        self.nodes.append(node)

    def get_random_node(self):
        return self.nodes[random.randint(0, len(self.nodes) - 1)]


class InputLayer(BaseLayer):
    """
    Input layer
    """
    def __init__(self, layer_id, nodes=[]):
        super().__init__(layer_id, nodes)


class HiddenLayer(BaseLayer):
    """
    Hidden layer
    """
    def __init__(self, layer_id, nodes=[]):
        super().__init__(layer_id, nodes)


class OutputLayer(BaseLayer):
    """
    Output layer
    """
    def __init__(self, layer_id, nodes=[]):
        super().__init__(layer_id, nodes)
