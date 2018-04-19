import torch
import numpy as np


def accuracy(y_pred, y_true):
    right = np.sum((y_pred.data.numpy() > 0.5) == y_true.data.numpy())
    total = y_true.data.numpy().shape[0]
    return 100. * float(right) / float(total)
