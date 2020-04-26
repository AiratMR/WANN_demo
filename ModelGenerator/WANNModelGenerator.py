import math
import random
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, log_loss
from joblib import Parallel, delayed
import multiprocessing as mp

from typing import List
from Model.WANNModel import WANNModel

EVAL_WEIGHTS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
num_cores = mp.cpu_count()


def generate_wann_model(x_train, y_train,
                        tol: float = 0.1,
                        niter: int = 50,
                        gen_num: int = 9,
                        sort_type: str = 'all',
                        selection: str = 'tournament',
                        metric: str = "mse") -> WANNModel:
    """
    Generate the WANN model by train data
    :param x_train: input train values -> 2d np.ndarray
    :param y_train: output train values -> 2d np.ndarray
    :param tol: model accuracy -> float
    :param niter: number of iterations -> int
    :param gen_num: number of models in generation -> int
    :param sort_type: type of sorting of generation ('all', 'conn', 'nodes')
    :param selection: type of selection ('none', 'tournament', 'elite-tournament')
    :param metric: metric function type
    :return: winner -> WANNModel
    """

    assert x_train.shape[0] == y_train.shape[0], "'x_train' shape not equal of 'y_train shape"
    assert 0 < tol < 1, "'tol' must be greater than 0 and less than 1"
    assert niter > 0, "'niter' must be greater than 0"
    assert gen_num > 0, "'gen_num' must be greater than 0"

    sort_types = {
        "all": _get_models_avg_nodes_and_conn_count,
        "nodes": _get_models_avg_nodes_count,
        "conn": _get_models_avg_conn_count
    }

    metrics = {
        "mse": mean_squared_error,
        "log-loss": log_loss
    }

    # init first generation
    winners_num = math.ceil(gen_num / 2)
    generation = _init_first_generation({'x': x_train[0], 'y': y_train[0]}, gen_num)

    # init sorter and metric
    sorter = sort_types[sort_type]
    loss_func = metrics[metric]

    # init number of elite models
    elite_models_number = math.ceil(gen_num * 0.1)
    elite_models = []

    # init best model
    best_model = None

    for iteration in range(niter):
        rand = random.random()
        is_min_sort = True if rand < 0.8 else False
        models = Parallel(n_jobs=num_cores)(delayed(_get_models_avg_min)(x_train, y_train, loss_func, model)
                                            for model in generation) \
            if is_min_sort else \
            Parallel(n_jobs=num_cores)(delayed(sorter)(x_train, y_train, loss_func, model)
                                       for model in generation)

        # non-dominated sorting of generation
        fronts = _non_dominated_sorting(models)
        log = "Iteration #{0}:".format(iteration)
        logging.info(log)
        print(log)
        for front in fronts:
            for i in front:
                log = "     Model #{0} - avg_mean squared error {1}".format(models[i][0], models[i][1])
                logging.info(log)
                print(log)
                if models[i][1] < tol:
                    return next(winner for winner in generation if winner.model_id == models[i][0])

        # selection of new candidates to new generation
        best_candidates = []
        for front in fronts:
            if len(best_candidates) + len(front) < winners_num:
                for i in front:
                    best_candidates.append(models[i])
            else:
                crowding_distance = _crowding_distance(models, front)
                distance_to_model = [(crowding_distance[i], models[index]) for i, index in enumerate(front)]
                sorted_front = sorted(distance_to_model, key=lambda it: (it[1], it[0]), reverse=True)
                for item in sorted_front:
                    best_candidates.append(item[1])

                    if len(best_candidates) == winners_num:
                        break
            if len(best_candidates) >= winners_num:
                break

        # exit if iteration is last
        if iteration == niter - 1:
            winners = [model[0] for model in best_candidates]
            winners = [next(model for model in generation if model.model_id == candidate) for candidate in
                       winners]
            best_model = _get_winner(x_train, y_train, loss_func, winners)
            break

        # init and modification of elite models
        if selection == 'elite-tournament':
            if iteration == 0:
                for i in range(elite_models_number):
                    elite_models.append((next(model.get_copy() for model in generation if
                                              model.model_id == best_candidates[i][0]), best_candidates[i][1]))
            else:
                for i in range(elite_models_number):
                    elite_errors = {elite_model[1] for elite_model in elite_models}
                    if best_candidates[i][1] not in elite_errors and \
                            best_candidates[i][1] < elite_models[i][1]:
                        elite_models[i] = (next(model.get_copy() for model in generation if
                                                model.model_id == best_candidates[i][0]), best_candidates[i][1])

        new_generation = []

        # binary tournament selection with elite selection
        if selection == 'elite-tournament':
            for elite in elite_models:
                new_generation.append(elite[0])
            while len(new_generation) != gen_num:
                candidates = np.random.choice(range(len(best_candidates)), 2)
                candidate1 = best_candidates[candidates[0]]
                candidate2 = best_candidates[candidates[1]]
                winner = candidate1 if candidate1[1] < candidate2[1] else candidate2
                new_generation.append(next(model.get_copy() for model in generation if model.model_id == winner[0]))

        # binary tournament selection
        elif selection == "tournament":
            while len(new_generation) != gen_num:
                candidates = np.random.choice(range(len(best_candidates)), 2)
                candidate1 = best_candidates[candidates[0]]
                candidate2 = best_candidates[candidates[1]]
                winner = candidate1 if candidate1[1] < candidate2[1] else candidate2
                new_generation.append(next(model.get_copy() for model in generation if model.model_id == winner[0]))

        # Just Pareto sorting
        else:
            while len(new_generation) != gen_num:
                for candidate in best_candidates:
                    new_generation.append(next(model.get_copy() for model in generation
                                               if model.model_id == candidate[0]))
                    if len(new_generation) == gen_num:
                        break

        for model in new_generation:
            model.random_mutation()

        generation = new_generation

    return best_model


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


def _get_models_avg_nodes_count(x_train, y_train, metric, model):
    """
    Evaluate and sort models by error and nodes count to increase
    :param x_train: input values -> 2d np.ndarray
    :param y_train: output values -> 2d np.ndarray
    :param metric: metric function
    :param model: WANN model -> WANNModel
    :return: list of sorted models by error -> List[Tuple(float, WANNModel)]
    """

    errors_avg = 0
    for i, value in enumerate(EVAL_WEIGHTS):
        model.set_weight(value)
        eval_result = model.evaluate_model(x_train)
        error = metric(y_train, eval_result)
        errors_avg += error

    return model.model_id, errors_avg / len(EVAL_WEIGHTS), model.nodes_count


def _get_models_avg_conn_count(x_train, y_train, metric, model):
    """
    Evaluate and sort models by error and connections count to increase
    :param x_train: input values -> 2d np.ndarray
    :param y_train: output values -> 2d np.ndarray
    :param metric: metric function
    :param model: WANN model -> WANNModel
    :return: list of sorted models by error -> List[Tuple(float, WANNModel)]
    """
    errors_avg = 0
    for i, value in enumerate(EVAL_WEIGHTS):
        model.set_weight(value)
        eval_result = model.evaluate_model(x_train)
        error = metric(y_train, eval_result)
        errors_avg += error

    return model.model_id, errors_avg / len(EVAL_WEIGHTS), model.connections_count


def _get_models_avg_min(x_train, y_train, metric, model):
    """
    Evaluate and sort models by average error and best error to increase
    :param x_train: input values -> 2d np.ndarray
    :param y_train: output values -> 2d np.ndarray
    :param metric: metric function
    :param model: WANN model -> WANNModel
    :return: list of sorted models by error -> List[Tuple(float, WANNModel)]
    """

    errors_avg = 0
    best_min = 999999999
    for i, value in enumerate(EVAL_WEIGHTS):
        model.set_weight(value)
        eval_result = model.evaluate_model(x_train)
        error = metric(y_train, eval_result)
        if error < best_min:
            best_min = error
        errors_avg += error

    return model.model_id, errors_avg / len(EVAL_WEIGHTS), best_min


def _get_models_avg_nodes_and_conn_count(x_train, y_train, metric, model):
    """
    Evaluate and sort models by error, nodes and connections count to increase
    :param x_train: input values -> 2d np.ndarray
    :param y_train: output values -> 2d np.ndarray
    :param metric: metric function
    :param model: WANN model -> WANNModel
    :return: list of sorted models by error -> List[Tuple(float, WANNModel)]
    """
    errors_avg = 0
    for i, value in enumerate(EVAL_WEIGHTS):
        model.set_weight(value)
        eval_result = model.evaluate_model(x_train)
        error = metric(y_train, eval_result)
        errors_avg += error

    return model.model_id, errors_avg / len(EVAL_WEIGHTS), model.nodes_count, model.connections_count


def _get_winner(x_train, y_train, metric, generation):
    """
    Get best model from generation
    :param x_train: input values -> 2d np.ndarray
    :param y_train: output values -> 2d np.ndarray
    :param metric: metric function
    :param generation: list with first generation of models -> List[WANNModel]
    :return: list of sorted models by error -> List[Tuple(float, WANNModel)]
    """
    best_model = generation[0]
    best_min = 999999999
    for model in generation:
        for i, value in enumerate(EVAL_WEIGHTS):
            model.set_weight(value)
            eval_result = model.evaluate_model(x_train)
            error = metric(y_train, eval_result)
            if error < best_min:
                best_min = error
                best_model = model.get_copy()
                best_model.set_weight(value)

    return best_model


def _non_dominated_sorting(models):
    S = [[] for _ in range(0, len(models))]
    front = [[]]
    n = [0 for _ in range(0, len(models))]
    rank = [0 for _ in range(0, len(models))]

    def and_operation(model1, model2):
        for i in range(1, len(model1)):
            if model1[i] <= model2[i]:
                continue
            else:
                return False
        return True

    def or_operation(model1, model2):
        for i in range(1, len(model1)):
            if model1[i] < model2[i]:
                return True
        return False

    for p in range(0, len(models)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(models)):
            if and_operation(models[p], models[q]) and or_operation(models[p], models[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif and_operation(models[q], models[p]) and or_operation(models[q], models[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)
    del front[len(front) - 1]
    return front


def _sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        min_index = np.argwhere(values == np.min(values)).flatten()[0]
        if min_index in list1:
            sorted_list.append(min_index)
        values[min_index] = 99999999
    return sorted_list


def _crowding_distance(models, front):
    distance = [0 for i in range(0, len(front))]
    distance[0] = 99999999
    distance[len(front) - 1] = 99999999
    for i in range(1, len(models[0])):
        values = [model[i] for model in models]
        sorted_values = _sort_by_values(front, values)
        for k in range(1, len(front) - 1):
            try:
                distance[k] = distance[k] + (values[sorted_values[k + 1]] - values[sorted_values[k - 1]]) / (
                        max(values) - min(values))
            except ZeroDivisionError:
                distance[k] = 99999999

    return distance
