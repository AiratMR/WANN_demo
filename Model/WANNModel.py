import random
import uuid
import numpy as np
import pickle
from pathlib import Path
from copy import deepcopy
from scipy.optimize import dual_annealing
from sklearn.metrics import mean_squared_error

from .Layers import InputLayer, HiddenLayer, OutputLayer
from .Nodes import InputNode, HiddenNode, OutputNode
from .Connections import Connection
from Utils import get_random_function


class WANNModel:
    def __init__(self, model_id, layers, weight):
        self.model_id = model_id
        self.layers = layers
        self.weight = weight
        self.nodes_count = 0
        self.connections_count = 0
        self.nodes = []

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
        nodes_count = 0
        connections_count = 0
        nodes = []
        # input layer build
        input_layer = InputLayer(layer_id=str(uuid.uuid4()))
        for _ in train_data['x']:
            node = InputNode(
                node_id=str(uuid.uuid4()),
                layer_id=input_layer.layer_id
            )
            input_layer.add_node(node=node)
            nodes.append(node)
            nodes_count += 1

        # output layer build
        output_layer = OutputLayer(layer_id=str(uuid.uuid4()))
        for _ in train_data['y']:
            node = OutputNode(
                node_id=str(uuid.uuid4()),
                layer_id=output_layer.layer_id,
                activation=get_random_function()
            )
            output_layer.add_node(node=node)
            nodes.append(node)
            nodes_count += 1

        model = WANNModel(model_id=str(uuid.uuid4()),
                          layers=[input_layer, output_layer],
                          weight=1)
        model.nodes_count = nodes_count
        model.nodes = nodes

        # add connections
        for node in output_layer.nodes:
            connection = Connection(node=input_layer.get_random_node(),
                                    weight=model.weight)
            node.prev_connections.add_connection(connection)
            connections_count += 1
        model.connections_count = connections_count

        return model

    def evaluate_model(self, input_data):
        """
        Evaluate model by input data
        @param input_data: 2d np.ndarray
        @return: result -> 2d np.ndarray
        """
        result = []

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
        if with_input:
            return self.nodes
        return [node for node in self.nodes if not isinstance(node, InputNode)]

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

        res = dual_annealing(obj_func, np.array([(-2, 2)]))
        self.weight = res.x[0]

    def random_mutation(self):
        """
        Random mutation of model
        """

        mutation_list = [
            self._change_activation_function,
            self._add_connection,
            self._add_new_node
        ]

        mutation = random.choice(mutation_list)
        mutation()
        print("""Random mutation - {0};
                 connections count = {1}
                 nodes count = {2}""".format(mutation.__name__, self.connections_count, self.nodes_count))

    def save(self, filename):
        """
        Save model object to file
        :param  filename: name of file to save
        """
        with(open(filename + ".pkl", 'wb')) as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath):
        """
        Load model from file
        :param filepath: path to file
        :return: WANNModel
        """
        path = Path(filepath)

        if path.exists():
            with(open(path, 'rb')) as file:
                return pickle.load(file)
        else:
            raise FileExistsError("File {0} does not exist".format(filepath))

    def _change_activation_function(self):
        """
        Change сurrent activation function of random node in model
        """
        nodes = self.get_all_nodes()
        node = random.choice(nodes)
        node.activation = get_random_function()

    def _get_random_layer_order(self, layer_id1, layer_id2):
        """
        Get order of two random layers
        :param layer_id1: first layer guid -> str
        :param layer_id2: second layer guid -> str
        :return: List[str]
        """
        ordered_layers_id = []
        for layer in self.layers:
            if layer.layer_id == layer_id1 or layer.layer_id == layer_id2:
                ordered_layers_id.append(layer.layer_id)
        return ordered_layers_id

    def _add_connection(self):
        """
        Add connection between random nodes
        """
        all_nodes = self.get_all_nodes(with_input=True)
        first_rand_node = random.choice(all_nodes)

        candidate_layers = [layer for layer in self.layers if layer.layer_id != first_rand_node.layer_id]
        second_rand_node = random.choice([node for layer in candidate_layers for node in layer.nodes])

        layer_id_to_node_map = {
            first_rand_node.layer_id: first_rand_node,
            second_rand_node.layer_id: second_rand_node
        }

        first_layer_id, second_layer_id = self._get_random_layer_order(first_rand_node.layer_id,
                                                                       second_rand_node.layer_id)

        prev_node = layer_id_to_node_map[first_layer_id]
        node = layer_id_to_node_map[second_layer_id]

        connection = Connection(prev_node, self.weight)
        if node.prev_connections.is_valid_connection(connection):
            node.prev_connections.add_connection(connection)
            self.connections_count += 1

    def _add_new_node(self):
        """
        Add new node between two random nodes
        """
        all_nodes = self.get_all_nodes()
        node = random.choice(all_nodes)
        prev_node = random.choice([connection.node for connection in node.prev_connections.connections])

        first_layer_index = next(i for i, layer in enumerate(self.layers) if layer.layer_id == prev_node.layer_id)
        second_layer_index = next(i for i, layer in enumerate(self.layers) if layer.layer_id == node.layer_id)
        layers_between = self.layers[first_layer_index:second_layer_index]

        if len(layers_between) == 1:
            self.layers.insert(first_layer_index + 1, HiddenLayer(str(uuid.uuid4())))
            new_layer = self.layers[first_layer_index + 1]

            new_node = HiddenNode(str(uuid.uuid4()), new_layer.layer_id, get_random_function())
            new_node.prev_connections.add_connection(Connection(prev_node, self.weight))

            new_layer.add_node(new_node)

            node.prev_connections.update_connection(prev_node, new_node)
            self.nodes.append(new_node)
        else:
            exist_layer = self.layers[first_layer_index + 1]

            new_node = HiddenNode(str(uuid.uuid4()), exist_layer.layer_id, get_random_function())
            new_node.prev_connections.add_connection(Connection(prev_node, self.weight))

            exist_layer.add_node(new_node)

            node.prev_connections.update_connection(prev_node, new_node)
            self.nodes.append(new_node)
        self.nodes_count += 1
        self.connections_count += 1
