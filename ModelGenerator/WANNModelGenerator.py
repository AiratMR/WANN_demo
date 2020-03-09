from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error

from typing import List
from Model import WANNModel

EVAL_WEIGHTS = [-2.0, -1.0, 0.1, 1.0, 2.0]


def generate_wann_model(x_train, y_train, tol: float = 0.1, niter: int = 50, gen_num: int = 5) -> WANNModel:
    x_scaled = minmax_scale(x_train)
    y_scaled = minmax_scale(y_train)
    generation = _init_first_generation({'x': x_train[0], 'y': y_train[0]}, gen_num)
    result = _order_models_by_error(x_scaled, y_scaled, generation)
    return generation


def _init_first_generation(train_data: dict, gen_num: int) -> List[WANNModel]:
    generation = []
    for _ in range(gen_num):
        generation.append(WANNModel.create_model(train_data))
    return generation


def _order_models_by_error(x_train, y_train, generation):
    model_result = {}
    for model in generation:
        errors_sum = 0
        for i, value in enumerate(EVAL_WEIGHTS):
            model.set_weight(value)
            eval_result = model.evaluate_model(x_train)
            error = mean_squared_error(y_train, eval_result)
            errors_sum += error
        model_result[model.model_id] = errors_sum
    return model_result
