import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

pd.options.display.max_columns = 999
pd.options.display.max_rows = 20

import models
import preprocessing
import train_test as tt
import error_plot as ep
import models_comp as mc

if __name__ == '__main__':
    data = preprocessing.load_data("diabetes")
    row, col = data.shape
    # X_train_tor, X_val_tor, X_test_tor, y_train_tor, y_val_tor, y_test_tor = preprocessing.seperate_data(
    #     data, holdout_split=0.2)
    cv_data, X_test_tor, y_test_tor = preprocessing.seperate_data(
        data, holdout_split=0.2)
    input_size = col - 1
    # *

    # print('\nNet: ')
    # tt.train_test(models.Net(input_size), X_train_tor,
    #               X_test_tor, y_train_tor, y_test_tor)
    # print('\nResNet: ')
    # tt.train_test(models.ResNet(input_size), X_train_tor,
    #               X_test_tor, y_train_tor, y_test_tor)
    # print('\nTDRNN: ')
    # tt.train_test(models.TDRNN(input_size), X_train_tor,
    #               X_test_tor, y_train_tor, y_test_tor)
    # print('\nODRNN: ')
    # tt.train_test(models.ODRNN(input_size), X_train_tor,
    #               X_test_tor, y_train_tor, y_test_tor)
    print('\nSORNN: ')
    # tt.train_test(models.SORNN(input_size), X_train_tor,
    #               X_test_tor, y_train_tor, y_test_tor)

    # tt.train_test(models.SORNN(input_size), cv_data, X_test_tor, y_test_tor)

    # ep.error_plot(models.SORNN(input_size), cv_data,
    #               X_test_tor, y_test_tor, iterations=10)

    network_list = [models.ResNet(input_size), models.TDRNN(input_size), models.ODRNN(input_size), models.SORNN(input_size)]
    name_network = ['ResNet', 'TDRNN', 'ODRNN', 'SORNN']
    print(len(network_list))
    mc.models_comp(network_list, cv_data, X_test_tor, y_test_tor, name_network, iterations=2)
