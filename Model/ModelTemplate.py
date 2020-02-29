import uuid

from .Layers import InputLayer, HiddenLayer, OutputLayer
from .Nodes import InputNode, HiddenNode, OutputNode
from .Connections import Connection
from Utils import FUNCTIONS, get_random_function


class ModelTemplate:
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
        model = ModelTemplate(layers=[input_layer, output_layer],
                              weight=1)

        for node in output_layer.nodes:
            connection = Connection(node=input_layer.get_random_node(),
                                    weight=model.weight)
            node.prev_connections.add_connection(connection)

        return model
