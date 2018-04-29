import numpy as np
import torch
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torch.autograd import Variable
import re


def load_data(dataset):
    data = preprocessor_libsvm_data(dataset)
    row, col = data.shape
    data = pd.DataFrame(data)
    data[[col - 1]] = data[[col - 1]].astype('int')
    return data


def preprocessor_libsvm_data(filename, format_label_func=lambda _: _):
    with open('./uci/' + filename + '.data', 'r') as inputfile:
        features = []
        labels = []
        for line in inputfile:
            container = line.rstrip().split()
            label = float(container[0])
            label = int(format_label_func(label))
            del container[0]
            pattern = re.compile(r"[-+]?\d+:([-+]?\d*\.\d+|[-+]?\d+)")
            feature = []
            for phrase in container:
                target = re.findall(pattern, phrase)
                feature.append(float(target[0]))
            features.append(feature)
            labels.append(label)
        classes = list(set(labels))
        for i in range(len(labels)):
            if labels[i] == classes[0]:
                labels[i] = 1
            else:
                labels[i] = 0
        features = np.array(features)
        labels = np.array(labels).reshape((-1, 1))
        labels = labels.astype(np.float)
        data = np.concatenate((features, labels), axis=1)
        return data


def show_data(data):
    pd.DataFrame.hist(data=data, figsize=[15, 15], bins=20)


def seperate_data(data, holdout_split=0.2):
    row, col = data.shape
    X_data = data.loc[:, :col - 2].values
    y_data = data.loc[:, col - 1].values

    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size=holdout_split,
                                                        shuffle=True)

    # transform to column vectors
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    print(('Original data points: {}'.format(np.shape(X_data)[0])))
    print(('Test split: {}%'.format(holdout_split * 100)))
    print(('Training points: {}, Test points: {}'.format(np.shape(X_train)[0],
                                                         np.shape(X_test)[0])))

    # Precompute normalization of k-fold train/val sets for inputs
    # Use scale by median and IQR (aka 'robust scaling')
    # Leave outputs as binary

    K_FOLDS = 5
    cv_data = [[]] * K_FOLDS
    index = 0

    kf = KFold(n_splits=K_FOLDS)
    for train_loc, val_loc in kf.split(X_train):
        # create robust scaler for input
        scale_in = RobustScaler().fit(X_train[train_loc])

        # train data for each fold
        X_train_scale = scale_in.transform(X_train[train_loc])
        y_train_scale = y_train[train_loc]  # don't scale binary

        # validation data for each fold
        X_val_scale = scale_in.transform(X_train[val_loc])
        y_val_scale = y_train[val_loc]

        # store to unpack later
        cv_data[index] = (
            Variable(torch.FloatTensor(X_train_scale), requires_grad=False),
            Variable(torch.FloatTensor(X_val_scale), requires_grad=False),
            Variable(torch.FloatTensor(y_train_scale), requires_grad=False),
            Variable(torch.FloatTensor(y_val_scale), requires_grad=False)
        )

        index += 1

    print(('Number of folds: {}'.format(K_FOLDS)))
    print(('Train points: {}'.format(np.size(train_loc))))
    print(('Val points: {}'.format(np.size(val_loc))))

    # create robust scaler for input
    scale_in = RobustScaler().fit(X_train)

    # all of test data
    X_test_scale = scale_in.transform(X_test)
    y_test_scale = y_test

    # prepare for PyTorch
    X_test_tor = Variable(torch.FloatTensor(X_test_scale), requires_grad=False)
    y_test_tor = Variable(torch.FloatTensor(y_test_scale), requires_grad=False)

    print('Test points: {}'.format(np.size(y_test_scale)))

    return cv_data, X_test_tor, y_test_tor

    # # create robust scaler for input
    # scale_in = RobustScaler().fit(X_train)
    #
    # # all of train data
    # X_train_scale = scale_in.transform(X_train)
    # y_train_scale = y_train  # don't scale binary
    #
    # # all of test data
    # X_test_scale = scale_in.transform(X_test)
    # y_test_scale = y_test
    #
    # # prepare for PyTorch
    # X_train_tor = Variable(torch.FloatTensor(
    #     X_train_scale), requires_grad=False)
    # X_test_tor = Variable(torch.FloatTensor(X_test_scale), requires_grad=False)
    # y_train_tor = Variable(torch.FloatTensor(
    #     y_train_scale), requires_grad=False)
    # y_test_tor = Variable(torch.FloatTensor(y_test_scale), requires_grad=False)
    #
    # print('Train points: {}'.format(np.size(y_train_scale)))
    # print('Test points: {}'.format(np.size(y_test_scale)))
    #
    # return X_train_tor, X_val_scale, X_test_tor, y_train_tor, y_val_scale, y_test_tor
