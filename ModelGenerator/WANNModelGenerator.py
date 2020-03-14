import numpy as np
import uuid
from sklearn.metrics import mean_squared_error
import random
from copy import deepcopy

from typing import List
from Model import WANNModel
from Model.Nodes import HiddenNode
from Model.Layers import HiddenLayer
from Model.Connections import Connection
from Utils import get_random_function

EVAL_WEIGHTS = [-2.0, -1.0, 0.1, 1.0, 2.0]


def generate_wann_model(x_train, y_train, tol: float = 0.1, niter: int = 50, gen_num: int = 5) -> WANNModel:
    x_scaled = (x_train - np.min(x_train)) / np.ptp(x_train)
    y_scaled = (y_train - np.min(y_train)) / np.ptp(y_train)
    generation = _init_first_generation({'x': x_train[0], 'y': y_train[0]}, gen_num)
    result = _sort_models_by_error(x_scaled, y_scaled, generation)

    winner_model = next(model for model in generation if model.model_id == result[0][1])
    _add_new_node(winner_model)
    res = winner_model.evaluate_model(x_scaled)
    copy_model = deepcopy(winner_model)

    return generation


def _init_first_generation(train_data: dict, gen_num: int) -> List[WANNModel]:
    generation = []
    for _ in range(gen_num):
        generation.append(WANNModel.create_model(train_data))
    return generation


def _sort_models_by_error(x_train, y_train, generation):
    model_result = {}
    for model in generation:
        errors_sum = 0
        for i, value in enumerate(EVAL_WEIGHTS):
            model.set_weight(value)
            eval_result = model.evaluate_model(x_train)
            error = mean_squared_error(y_train, eval_result)
            errors_sum += error
        model_result[model.model_id] = errors_sum

    return sorted((value, key) for (key, value) in model_result.items())


def _change_activation_function(model):
    nodes = model.get_all_nodes()
    node = random.choice(nodes)
    node.activation = get_random_function()


def _add_connection(model):
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
    ordered_layers_id = []
    for layer in model.layers:
        if layer.layer_id == layer_id1 or layer.layer_id == layer_id2:
            ordered_layers_id.append(layer.layer_id)
    return ordered_layers_id


def _add_new_node(model):
    all_nodes = model.get_all_nodes()
    node = random.choice(all_nodes)
    prev_node = random.choice([connection.node for connection in node.prev_connections.connections])

    first_layer_index = next(i for i, layer in enumerate(model.layers) if layer.layer_id == prev_node.layer_id)
    second_layer_index = next(i for i, layer in enumerate(model.layers) if layer.layer_id == node.layer_id)
    layers_between = model.layers[first_layer_index:second_layer_index]

    if len(layers_between) == 1:
        model.layers.insert(first_layer_index + 1, HiddenLayer(str(uuid.uuid4())))
        new_layer = model.layers[first_layer_index + 1]

        new_node = HiddenNode(str(uuid.uuid4()), new_layer.layer_id, get_random_function())
        new_node.prev_connections.add_connection(Connection(prev_node, model.weight))

        new_layer.add_node(new_node)

        node.prev_connections.edit_connection(prev_node, new_node)
    else:
        exist_layer = model.layers[first_layer_index + 1]

        new_node = HiddenNode(str(uuid.uuid4()), exist_layer.layer_id, get_random_function())
        new_node.prev_connections.add_connection(Connection(prev_node, model.weight))

        exist_layer.add_node(new_node)

        node.prev_connections.edit_connection(prev_node, new_node)
