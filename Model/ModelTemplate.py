import uuid
import numpy as np

from .Layers import InputLayer, HiddenLayer, OutputLayer
from .Nodes import InputNode, HiddenNode, OutputNode
from .Connections import Connection
from Utils import FUNCTIONS, get_random_function


class WANNModel:
    def __init__(self, layers, weight):
        self.layers = layers
        self.weight = weight

    @staticmethod
    def create_model(input_data):
        """
        Generate model by input data
        @param input_data:
               {
                  'x': 1d array,
                  'y': 1d array
               }
        @return: WANN model
        """
        # input layer build
        input_layer = InputLayer(layer_id=uuid.uuid4())
        for _ in input_data['x']:
            input_layer.add_node(node=InputNode(
                node_id=uuid.uuid4(),
                layer_id=input_layer.layer_id
            ))

        # output layer build
        output_layer = OutputLayer(layer_id=uuid.uuid4())
        for _ in input_data['y']:
            output_layer.add_node(node=OutputNode(
                node_id=uuid.uuid4(),
                layer_id=output_layer.layer_id,
                activation=get_random_function()
            ))

        # ToDo - генерировать изначальный вес
        model = WANNModel(layers=[input_layer, output_layer],
                          weight=1)

        for node in output_layer.nodes:
            connection = Connection(node=input_layer.get_random_node(),
                                    weight=model.weight)
            node.prev_connections.add_connection(connection)

        return model

    def evaluate_model(self, input_data):
        """
        Evaluate model by input data
        @param input_data: 2d np.ndarray
        @return: result -> 2d np.ndarray
        """
        result = []
        # ToDo - сделать валидацию входных данных
        for data in input_data:
            for layer in self.layers:
                for i, node in enumerate(layer.nodes):
                    # init input data values in input layer
                    if isinstance(layer, InputLayer):
                        node.input_value = data[i]

                    node.evaluate_node_output()

                # create one result
                if isinstance(layer, OutputLayer):
                    result.append([res.value for res in layer.nodes])

        return np.array(result)
