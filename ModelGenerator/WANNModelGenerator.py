import numpy as np
import uuid
from sklearn.metrics import mean_squared_error
import random

from typing import List
from Model.WANNModel import WANNModel
from Model.Nodes import HiddenNode
from Model.Layers import HiddenLayer
from Model.Connections import Connection
from Utils import get_random_function

EVAL_WEIGHTS = [-2.0, -1.0, 0.1, 1.0, 2.0]


# ToDo - оформить все это в класс
def generate_wann_model(x_train, y_train,
                        tol: float = 0.1,
                        niter: int = 50,
                        gen_num: int = 9,
                        nwinners: int = 3) -> List[WANNModel]:
    """
    Generate the WANN model by train data
    :param x_train: input train values -> 2d np.ndarray
    :param y_train: output train values -> 2d np.ndarray
    :param tol: model accuracy -> float
    :param niter: number of iterations -> int
    :param gen_num: number of models in generation -> int
    :param nwinners: number of winners -> int
    :return: list of winners -> List[WANNModel]
    """

    assert x_train.shape[0] == y_train.shape[0], "'x_train' shape not equal of 'y_train shape"
    assert gen_num > nwinners, "Value of 'gen_num' must be greater than 'nwinners'"

    generation = _init_first_generation({'x': x_train[0], 'y': y_train[0]}, gen_num)

    for iteration in range(niter):
        iter_result = _sort_models_by_error(x_train, y_train, generation)

        print("Iteration #{0}:".format(iteration))
        for i, result in enumerate(iter_result):
            print("     Model #{0} - mean squared error {1}".format(i, result[0]))
            if result[0] < tol:
                return generation[:nwinners]

        winner_models = []
        for i in range(nwinners):
            winner_models.append(next(model for model in generation if model.model_id == iter_result[i][1]))

        for i in range(nwinners):
            for j in range(nwinners - 1):
                winner_models.append(winner_models[i].get_copy())
            if len(winner_models) == gen_num:
                break

        for i, model in enumerate(winner_models):
            modification = random.choice(MODIFICATION_LIST)
            print(modification.__name__)
            modification(model)

        generation = winner_models

    return generation[:nwinners]


def _init_first_generation(train_data: dict, gen_num: int) -> List[WANNModel]:
    """
    Initialization of first generation of models
    :param train_data: dict with structure
    {
        'x': 1d np.ndarray
        'y': 1d np.ndarray
    }
    :param gen_num: number of models in generation -> int
    :return: first generation list -> List[WANNModel]
    """
    generation = []
    for _ in range(gen_num):
        generation.append(WANNModel.create_model(train_data))
    return generation


def _sort_models_by_error(x_train, y_train, generation):
    """
    Evaluate and sort models by error to increase
    :param x_train: input values -> 2d np.ndarray
    :param y_train: output values -> 2d np.ndarray
    :param generation: list with first generation of models -> List[WANNModel]
    :return: list of sorted models by error -> List[Tuple(float, WANNModel)]
    """
    x_scaled = (x_train - np.min(x_train)) / np.ptp(x_train)
    y_min, y_ptp = np.min(y_train), np.ptp(y_train)

    def y_scaled(y):
        return y * y_ptp + y_min

    model_result = {}
    for model in generation:
        errors_sum = 0
        for i, value in enumerate(EVAL_WEIGHTS):
            model.set_weight(value)
            eval_result = model.evaluate_model(x_scaled)
            error = mean_squared_error(y_train, y_scaled(eval_result))
            errors_sum += error
        model_result[model.model_id] = errors_sum

    return sorted((value, key) for (key, value) in model_result.items())


def _change_activation_function(model):
    """
    Change сurrent activation function of random node in model
    :param model: WANN model -> WANNModel
    """
    nodes = model.get_all_nodes()
    node = random.choice(nodes)
    node.activation = get_random_function()


def _add_connection(model):
    """
    Add connection between random nodes
    :param model: WANN model -> WANNModel
    """
    all_nodes = model.get_all_nodes(with_input=True)
    first_rand_node = random.choice(all_nodes)

    candidate_layers = [layer for layer in model.layers if layer.layer_id != first_rand_node.layer_id]
    second_rand_node = random.choice([node for layer in candidate_layers for node in layer.nodes])

    layer_id_to_node_map = {
        first_rand_node.layer_id: first_rand_node,
        second_rand_node.layer_id: second_rand_node
    }

    first_layer_id, second_layer_id = _get_random_layer_order(first_rand_node.layer_id, second_rand_node.layer_id,
                                                              model)

    prev_node = layer_id_to_node_map[first_layer_id]
    node = layer_id_to_node_map[second_layer_id]

    connection = Connection(prev_node, model.weight)
    if node.prev_connections.is_valid_connection(connection):
        node.prev_connections.add_connection(connection)


def _get_random_layer_order(layer_id1, layer_id2, model):
    """
    Get order of two random layers
    :param layer_id1: first layer guid -> str
    :param layer_id2: second layer guid -> str
    :param model: WANN model -> WANNModel
    :return: List[str]
    """
    ordered_layers_id = []
    for layer in model.layers:
        if layer.layer_id == layer_id1 or layer.layer_id == layer_id2:
            ordered_layers_id.append(layer.layer_id)
    return ordered_layers_id


def _add_new_node(model):
    """
    Add new node between two random nodes
    :param model: WANN model -> WANNModel
    """
    all_nodes = model.get_all_nodes(with_input=True)
    first_rand_node = random.choice(all_nodes)

    candidate_layers = [layer for layer in model.layers if layer.layer_id != first_rand_node.layer_id]
    second_rand_node = random.choice([node for layer in candidate_layers for node in layer.nodes])

    layer_id_to_node_map = {
        first_rand_node.layer_id: first_rand_node,
        second_rand_node.layer_id: second_rand_node
    }

    first_layer_id, second_layer_id = _get_random_layer_order(first_rand_node.layer_id, second_rand_node.layer_id,
                                                              model)

    prev_node = layer_id_to_node_map[first_layer_id]
    node = layer_id_to_node_map[second_layer_id]

    first_layer_index = next(i for i, layer in enumerate(model.layers) if layer.layer_id == prev_node.layer_id)
    second_layer_index = next(i for i, layer in enumerate(model.layers) if layer.layer_id == node.layer_id)
    layers_between = model.layers[first_layer_index:second_layer_index]

    if len(layers_between) == 1:
        model.layers.insert(first_layer_index + 1, HiddenLayer(str(uuid.uuid4())))
        new_layer = model.layers[first_layer_index + 1]

        new_node = HiddenNode(str(uuid.uuid4()), new_layer.layer_id, get_random_function())
        new_node.prev_connections.add_connection(Connection(prev_node, model.weight))

        new_layer.add_node(new_node)

        node.prev_connections.update_connection(prev_node, new_node)
    else:
        exist_layer = model.layers[first_layer_index + 1]

        new_node = HiddenNode(str(uuid.uuid4()), exist_layer.layer_id, get_random_function())
        new_node.prev_connections.add_connection(Connection(prev_node, model.weight))

        exist_layer.add_node(new_node)

        node.prev_connections.update_connection(prev_node, new_node)


# List of model modifications
MODIFICATION_LIST = [_change_activation_function,
                     _add_connection,
                     _add_new_node]
