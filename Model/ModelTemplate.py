import uuid
import numpy as np

from .Layers import InputLayer, HiddenLayer, OutputLayer
from .Nodes import InputNode, HiddenNode, OutputNode
from .Connections import Connection
from Utils import get_random_function


class WANNModel:
    def __init__(self, model_id, layers, weight):
        self.model_id = model_id
        self.layers = layers
        self.weight = weight

    @classmethod
    def create_model(cls, train_data):
        """
        Generate model by input data
        @param train_data:
               {
                  'x': 1d array,
                  'y': 1d array
               }
        @return: WANN model
        """
        # input layer build
        input_layer = InputLayer(layer_id=str(uuid.uuid4()))
        for _ in train_data['x']:
            input_layer.add_node(node=InputNode(
                node_id=str(uuid.uuid4()),
                layer_id=input_layer.layer_id
            ))

        # output layer build
        output_layer = OutputLayer(layer_id=str(uuid.uuid4()))
        for _ in train_data['y']:
            output_layer.add_node(node=OutputNode(
                node_id=str(uuid.uuid4()),
                layer_id=output_layer.layer_id,
                activation=get_random_function()
            ))

        # ToDo - генерировать изначальный вес
        model = WANNModel(model_id=str(uuid.uuid4()),
                          layers=[input_layer, output_layer],
                          weight=1)

        # add connections
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

    def set_weight(self, value):
        self.weight = value

        for layer in self.layers:
            for node in layer.nodes:
                if not isinstance(layer, InputLayer):
                    for conn in node.prev_connections.connections:
                        conn.weight = value
