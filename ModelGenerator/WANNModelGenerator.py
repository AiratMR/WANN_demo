import math
import numpy as np
from sklearn.metrics import mean_squared_error

from typing import List
from Model.WANNModel import WANNModel

EVAL_WEIGHTS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

"""
ToDo - реализовать ранжирование по: 1) В 80% по средней ошибке и количеству нейронов (связей);
                                    2) В 20% по средней ошибке и ошибке одного лучшего веса;
Сначала формируем ранги по алгоритму "Недоминируемой сортировки";
Затем формируем из рангов кандидатов на новое поколение;
Если n-ый ранг не польностью входит в новое поколение, то сортируем его критерию разреженности, которая
строится на основе вычисленного Манхетенского расстояния для каждого элемента до его левого и правого соседа
(расстояние для крайних элементов считать бесконечным)
"""


def generate_wann_model(x_train, y_train,
                        tol: float = 0.1,
                        niter: int = 50,
                        gen_num: int = 9) -> WANNModel:
    """
    Generate the WANN model by train data
    :param x_train: input train values -> 2d np.ndarray
    :param y_train: output train values -> 2d np.ndarray
    :param tol: model accuracy -> float
    :param niter: number of iterations -> int
    :param gen_num: number of models in generation -> int
    :return: winner -> WANNModel
    """

    assert x_train.shape[0] == y_train.shape[0], "'x_train' shape not equal of 'y_train shape"
    winners_num = math.ceil(gen_num / 2)
    generation = _init_first_generation({'x': x_train[0], 'y': y_train[0]}, gen_num)

    for iteration in range(niter):
        models = _get_models_avg_nodes_count(x_train, y_train, generation)
        fronts = _non_dominated_sorting(models)

        print("Iteration #{0}:".format(iteration))
        for front in fronts:
            for i in front:
                print("     Model #{0} - avg_mean squared error {1}".format(models[i][0], models[i][1]))
                if models[i][1] < tol:
                    return next(winner for winner in generation if winner.model_id == models[i][0])

        new_generation = []
        while len(new_generation) != winners_num:
            for front in fronts:
                if len(new_generation) + len(front) < winners_num:
                    for i in front:
                        new_generation.append(next(model for model in generation if model.model_id == models[i][0]))
                else:
                    crowding_distance = _crowding_distance(models, front)
                    distance_to_model = {models[index][0]: crowding_distance[i] for i, index in enumerate(front)}
                    sorted_front = sorted((value, key) for key, value in distance_to_model.items())
                    sorted_front.reverse()
                    for item in sorted_front:
                        new_generation.append(next(model for model in generation if model.model_id == item[1]))

                        if len(new_generation) == winners_num:
                            break
                    break

        for i in range(winners_num):
            new_generation.append(new_generation[i].get_copy())
            if len(new_generation) == gen_num:
                break

        for model in new_generation:
            model.random_mutation()

        generation = new_generation

    return generation[0]


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


def _get_models_avg_nodes_count(x_train, y_train, generation):
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

    model_result = []
    for model in generation:
        errors_avg = 0
        for i, value in enumerate(EVAL_WEIGHTS):
            model.set_weight(value)
            eval_result = model.evaluate_model(x_scaled)
            error = mean_squared_error(y_train, y_scaled(eval_result))
            errors_avg += error
        model_result.append((model.model_id, errors_avg / len(EVAL_WEIGHTS), model.nodes_count))

    return model_result


def _non_dominated_sorting(models):
    S = [[] for i in range(0, len(models))]
    front = [[]]
    n = [0 for i in range(0, len(models))]
    rank = [0 for i in range(0, len(models))]

    for p in range(0, len(models)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(models)):
            if (models[p][1] <= models[q][1] and models[p][2] <= models[q][2]) and (
                    models[p][1] < models[q][1] or models[p][2] < models[q][2]):
                if q not in S[p]:
                    S[p].append(q)
            elif (models[q][1] <= models[p][1] and models[q][2] <= models[p][2]) and (
                    models[q][1] < models[p][1] or models[q][2] < models[p][2]):
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


def _index_locator(a, arr):
    for i in range(0, len(arr)):
        if arr[i] == a:
            return i
    return -1


def _sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if _index_locator(np.min(values), values) in list1:
            sorted_list.append(_index_locator(np.min(values), values))
        values[_index_locator(np.min(values), values)] = 99999999
    return sorted_list


def _crowding_distance(models, front):
    distance = [0 for i in range(0, len(front))]
    values1 = [model[1] for model in models]
    values2 = [model[2] for model in models]

    sorted1 = _sort_by_values(front, values1)
    sorted2 = _sort_by_values(front, values2)

    distance[0] = 99999999
    distance[len(front) - 1] = 99999999
    for k in range(1, len(front) - 1):
        try:
            distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (
                    max(values1) - min(values1))
        except ZeroDivisionError:
            distance[k] = 99999999
    for k in range(1, len(front) - 1):
        try:
            distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                    max(values2) - min(values2))
        except ZeroDivisionError:
            distance[k] = 99999999
    return distance
