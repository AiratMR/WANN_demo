import numpy as np

from Model import WANNModel

model = WANNModel.create_model({
    'x': [1, 2, 3],
    'y': [1, 2]
})

input_data = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
result = model.evaluate_model(input_data)

a = 1
