import numpy as np
import pandas as pd
import logging

from Model import WANNModel
from keras.datasets import boston_housing
from ModelGenerator import generate_wann_model

logging.basicConfig(filename="wann.log", level=logging.INFO)

if __name__ == "__main__":
    # Cu - dataset
    data = pd.read_excel('data.xlsx')
    data = np.array(data)

    input_data = data[:, :4]
    in_scaled = (input_data - np.min(input_data)) / np.ptp(input_data)
    output_data = data[:, 4:]
    out_scaled = (output_data - np.min(output_data)) / np.ptp(output_data)

    logging.info("Cu - generating model start")

    logging.info("Cu - min connections optimization:")
    result = generate_wann_model(in_scaled, out_scaled, tol=0.001, gen_num=40, niter=512,
                                 sort_type="conn")
    model = result
    model.save('cu_model_conn')

    logging.info("Cu - min nodes optimization:")
    result = generate_wann_model(in_scaled, out_scaled, tol=0.001, gen_num=40, niter=512,
                                 sort_type="nodes")
    model = result
    model.save('cu_model_nodes')

    logging.info("Cu_vod - min connections and nodes optimization:")
    result = generate_wann_model(in_scaled, out_scaled, tol=0.001, gen_num=40, niter=512,
                                 sort_type="all")
    model = result
    model.save('cu_model_all')

    # Boston dataset
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    x_scaled = (x_train - np.min(x_train)) / np.ptp(x_train)
    y_scaled = (y_train - np.min(y_train)) / np.ptp(y_train)

    y_temp = []
    for i in y_scaled:
        y_temp.append([i])
    y_scaled = np.array(y_temp)

    logging.info("Boston - generating model start")

    logging.info("Boston - min connections optimization:")
    result = generate_wann_model(x_scaled, y_scaled, tol=0.001, gen_num=40, niter=512,
                                 sort_type="conn")
    model = result
    model.save('boston_model_conn')

    logging.info("Boston - min nodes optimization:")
    result = generate_wann_model(x_scaled, y_scaled, tol=0.001, gen_num=40, niter=512,
                                 sort_type="nodes")
    model = result
    model.save('boston_model_nodes')

    logging.info("Boston - min connections and nodes optimization:")
    result = generate_wann_model(x_scaled, y_scaled, tol=0.001, gen_num=40, niter=512,
                                 sort_type="all")
    model = result
    model.save('boston_model_all')
    #
    # model = WANNModel.load('cu_model_all.pkl')
    # model.train_weight(input_data, output_data)
