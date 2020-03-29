import numpy as np
import pandas as pd

from ModelGenerator import generate_wann_model

# model = WANNModel.create_model({
#     'x': [1, 2, 3],
#     'y': [1, 2]
# })

data = pd.read_excel('data.xlsx')
data = np.array(data)

input_data = data[:, :1]
output_data = data[:, 1:]
# result = model.evaluate_model(input_data)

result = generate_wann_model(input_data, output_data, tol=0.0001)
model = result[0]
model.train_weight(input_data, output_data)

a = 1
