import numpy as np

from ModelGenerator import generate_wann_model

# model = WANNModel.create_model({
#     'x': [1, 2, 3],
#     'y': [1, 2]
# })

input_data = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
output_data = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
# result = model.evaluate_model(input_data)

result = generate_wann_model(input_data, output_data)

a = 1
