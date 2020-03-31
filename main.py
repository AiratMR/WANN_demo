import numpy as np
import pandas as pd

from ModelGenerator import generate_wann_model

# model = WANNModel.create_model({
#     'x': [1, 2, 3],
#     'y': [1, 2]
# })

# data = pd.read_excel('data.xlsx')
# data = np.array(data)
#
# input_data = data[:, :1]
# output_data = data[:, 1:]
# result = model.evaluate_model(input_data)

input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
output_data = np.array([[2, 3], [4, 5], [6, 7]])

result = generate_wann_model(input_data, output_data, tol=0.0001, gen_num=50, niter=500)
model = result
model.train_weight(input_data, output_data)

a = 1
