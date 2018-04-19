import torch
import numpy as np
from aux import accuracy
import matplotlib.pyplot as plt
from torch.autograd import Variable


def train_test(network, X_train_tor, X_test_tor, y_train_tor, y_test_tor):

    VERBOSE = True

    final_net = network
    final_net.train()

    # optimizer = torch.optim.Adam(net.parameters(), weight_decay=final_net.L2_PEN, lr=final_net.LR)
    optimizer = torch.optim.SGD(final_net.parameters(),
                                weight_decay=final_net.L2_PEN,
                                lr=final_net.LR,
                                momentum=0.0,
                                dampening=0.0,
                                nesterov=False)
    loss_func = torch.nn.BCELoss()

    epochs = np.arange(final_net.NUM_EPOCHS)
    hist_loss_train = np.zeros(final_net.NUM_EPOCHS)
    hist_loss_test = np.zeros(final_net.NUM_EPOCHS)
    hist_correct_train = np.zeros(final_net.NUM_EPOCHS)
    hist_correct_test = np.zeros(final_net.NUM_EPOCHS)

    for step in range(final_net.NUM_EPOCHS):
        # Show improvement on test set
        final_net.eval()
        test_pred = final_net(X_test_tor)
        loss_test = loss_func(test_pred, y_test_tor)
        hist_loss_test[step] = loss_test
        hist_correct_test[step] = accuracy(test_pred, y_test_tor)

        # Perform train cost fn
        final_net.train()
        train_pred = final_net(X_train_tor)
        loss_train = loss_func(train_pred, y_train_tor)
        hist_loss_train[step] = loss_train
        hist_correct_train[step] = accuracy(train_pred, y_train_tor)

        if VERBOSE and ((step % 100) == 0) and step != 0:
            print('Step: {}'.format(step))
            print('\tTrain Accuracy: {}%'.format(hist_correct_train[step]))
            print('\tTest Accuracy: {}%'.format(hist_correct_test[step]))

        # Update based on train performance
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    print('\nFinal Train Accuracy: {}%'.format(hist_correct_train[-1]))
    print('Final Test Accuracy: {}%'.format(hist_correct_test[-1]))
    # print('Mean cross validation performance: {}'.format(fold_mean))

    plt.plot(epochs, hist_correct_train, epochs, hist_correct_test)
    # plt.title('Final Test Accuracy'.format(fold))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()

    # Our work
    X_train_tor_noise = X_train_tor + \
        Variable(torch.randn(X_train_tor.size()) * 1.0)
    final_net.eval()
    train_pred = final_net(X_train_tor_noise)
    loss_noise = accuracy(train_pred, y_train_tor)
    print('\nAccuracy after adding noise: {}'.format(loss_noise))
