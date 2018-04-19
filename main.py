import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing
from torch.nn import init
import math
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
import aux

if __name__ == '__main__':
    data = preprocessing.load_data("madelon")
    row, col = data.shape
    X_train_tor, X_test_tor, y_train_tor, y_test_tor = preprocessing.seperate_data(data, holdout_split=0.2)

    input_size = col - 1
    print('\nTDRNN: ')
    tt.train_test(models.TDRNN(input_size), X_train_tor, X_test_tor, y_train_tor, y_test_tor)
    print('\nNet: ')
    tt.train_test(models.Net(input_size), X_train_tor, X_test_tor, y_train_tor, y_test_tor)
    print('\nResNet: ')
    tt.train_test(models.ResNet(input_size), X_train_tor, X_test_tor, y_train_tor, y_test_tor)
