"""
The module contains the definition of the classes of layers of the neural network
"""
import numpy as np


class BaseLayer:
    """
    Base layer
    """
    def __init__(self, layer_id: int, nodes: np.ndarray):
        self._layer_id = layer_id
        self._nodes = nodes


class InputLayer(BaseLayer):
    """
    Input layer
    """
    def __init__(self, layer_id: int, nodes: np.ndarray):
        super().__init__(layer_id, nodes)


class HiddenLayer(BaseLayer):
    """
    Input layer
    """
    def __init__(self, layer_id: int, nodes: np.ndarray):
        super().__init__(layer_id, nodes)


class OutputLayer(BaseLayer):
    """
    Input layer
    """
    def __init__(self, layer_id: int, nodes: np.ndarray):
        super().__init__(layer_id, nodes)
