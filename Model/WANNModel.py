import uuid
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from .Layers import InputLayer, OutputLayer
from .Nodes import InputNode, OutputNode
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
        """
        Set new value of weight to all nodes in model
        :param value: new value -> float
        """
        self.weight = value

        for layer in self.layers:
            for node in layer.nodes:
                if not isinstance(layer, InputLayer):
                    for conn in node.prev_connections.connections:
                        conn.weight = value

    def get_all_nodes(self, with_input: bool = False):
        """
        Get list of all nodes
        :param with_input: True - get nodes from input layer too -> bool
        :return: list of nodes -> List[BaseNode]
        """
        nodes = []
        for layer in self.layers:
            if not with_input:
                if isinstance(layer, InputLayer):
                    continue
            for node in layer.nodes:
                nodes.append(node)
        return nodes

    def get_copy(self):
        """
        Return copy of model
        :return: copy of model -> WANNModel
        """
        copy_obj = deepcopy(self)
        copy_obj.model_id = str(uuid.uuid4())
        return copy_obj

    def train_weight(self, x_train, y_train):
        """
        Find the optimal value of weight
        :param x_train: input values
        :param y_train: output values
        """
        x_scaled = (x_train - np.min(x_train)) / np.ptp(x_train)
        y_min, y_ptp = np.min(y_train), np.ptp(y_train)

        def y_scaled(y):
            return y * y_ptp + y_min

        def obj_func(x):
            self.weight = x[0]
            self.set_weight(x[0])
            eval_result = self.evaluate_model(x_scaled)
            print("weight={0}".format(self.weight))
            print(y_scaled(eval_result))
            return mean_squared_error(y_train, y_scaled(eval_result))

        res = minimize(obj_func, np.array([1]), method='Nelder-Mead')
        self.weight = res.x[0]
