import torch
import numpy as np
from aux import accuracy
import matplotlib.pyplot as plt
from torch.autograd import Variable

VERBOSE = True
# def train_test(network, X_train_tor, X_val_to, X_test_tor, y_train_tor, y_test_tor):
def train_test(network, cv_data, X_test_tor, y_test_tor):
    K_FOLDS = 5

    for fold in range(K_FOLDS):
        print('Fold {} of {}'.format(fold+1, K_FOLDS))

        X_train_tor, X_val_tor, y_train_tor, y_val_tor = cv_data[fold]

        net = network
        net.train()

    #     optimizer = torch.optim.Adam(net.parameters(), weight_decay=net.L2_PEN, lr=net.LR)
        optimizer = torch.optim.SGD(net.parameters(),
                                    weight_decay=net.L2_PEN,
                                    lr=net.LR,
                                    momentum=0.00,
                                    dampening=0.00,
                                    nesterov=False)
        loss_func = torch.nn.BCELoss()

        epochs = np.arange(net.NUM_EPOCHS)
        hist_loss_train = np.zeros(net.NUM_EPOCHS)
        hist_loss_val = np.zeros(net.NUM_EPOCHS)
        hist_correct_train = np.zeros(net.NUM_EPOCHS)
        hist_correct_val = np.zeros(net.NUM_EPOCHS)

        for step in range(net.NUM_EPOCHS):
            # Show improvement on val set
            net.eval()
            val_pred = net(X_val_tor)
            loss_val = loss_func(val_pred, y_val_tor)
            hist_loss_val[step] = loss_val
            hist_correct_val[step] = accuracy(val_pred, y_val_tor)

            # Perform train cost fn
            net.train()
            train_pred = net(X_train_tor)
            loss_train = loss_func(train_pred, y_train_tor)
            hist_loss_train[step] = loss_train
            hist_correct_train[step] = accuracy(train_pred, y_train_tor)

            if VERBOSE and ((step % 100) == 0):
                print('Step: {}'.format(step))
                print('\tTrain Accuracy: {}'.format(hist_correct_train[step]))
                print('\tVal Accuracy: {}'.format(hist_correct_val[step]))

            # Update based on train performance
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        # accuracy_val_scores[fold] = hist_correct_val[-1]

        plt.plot(epochs, hist_correct_train, epochs, hist_correct_val)
        plt.title('Fold {} Accuracy'.format(fold+1))
        plt.xlabel('Epoch')
        plt.ylabel('Percent Accuracy')
        plt.ylim((0, 100))
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.show()

    # Accuracy of test set
    print(X_test_tor.norm)
    net.eval()
    test_pred = net(X_test_tor)
    loss_test = accuracy(test_pred, y_test_tor)
    print('\nAccuracy on test set: {0:.3f}%'.format(loss_test))



    # Test of stability of noise
    X_test_tor_noise = X_test_tor + \
        Variable(torch.randn(X_test_tor.size()) * 0.1)
    net.eval()
    test_pred = net(X_test_tor_noise)
    loss_noise = accuracy(test_pred, y_test_tor)
    print('\nAccuracy after adding noise: {0:.3f}%'.format(loss_noise))








    # VERBOSE = True
    #
    # final_net = network
    # final_net.train()
    #
    # # optimizer = torch.optim.Adam(final_net.parameters(), weight_decay=final_net.L2_PEN, lr=final_net.LR)
    # optimizer = torch.optim.SGD(final_net.parameters(),
    #                             weight_decay=final_net.L2_PEN,
    #                             lr=final_net.LR,
    #                             momentum=0.0,
    #                             dampening=0.0,
    #                             nesterov=False)
    # loss_func = torch.nn.BCELoss()
    #
    # epochs = np.arange(final_net.NUM_EPOCHS)
    # hist_loss_train = np.zeros(final_net.NUM_EPOCHS)
    # hist_loss_test = np.zeros(final_net.NUM_EPOCHS)
    # hist_correct_train = np.zeros(final_net.NUM_EPOCHS)
    # hist_correct_test = np.zeros(final_net.NUM_EPOCHS)
    #
    # for step in range(final_net.NUM_EPOCHS):
    #
    #     # Perform train cost fn
    #     final_net.train()
    #     train_pred = final_net(X_train_tor)
    #     loss_train = loss_func(train_pred, y_train_tor)
    #     hist_loss_train[step] = loss_train
    #     hist_correct_train[step] = accuracy(train_pred, y_train_tor)
    #
    #     # Show improvement on test set
    #     final_net.eval()
    #     test_pred = final_net(X_test_tor)
    #     loss_test = loss_func(test_pred, y_test_tor)
    #     hist_loss_test[step] = loss_test
    #     hist_correct_test[step] = accuracy(test_pred, y_test_tor)
    #
    #     if VERBOSE and ((step % 100) == 0) and step != 0:
    #         print('Step: {}'.format(step))
    #         print('\tTrain Accuracy: {0:.3f}%'.format(hist_correct_train[step]))
    #         print('\tTest Accuracy: {0:.3f}%'.format(hist_correct_test[step]))
    #
    #     # Update based on train performance
    #     optimizer.zero_grad()
    #     loss_train.backward()
    #     optimizer.step()
    #
    # print('\nFinal Train Accuracy: {0:.3f}%'.format(hist_correct_train[-1]))
    # print('Final Test Accuracy: {0:.3f}%'.format(hist_correct_test[-1]))
    # # print('Mean cross validation performance: {}'.format(fold_mean))
    #
    # plt.plot(epochs, hist_correct_train, epochs, hist_correct_test)
    # # plt.title('Final Test Accuracy'.format(fold))
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(['Train', 'Test'], loc='lower right')
    # plt.show()
    #
    # # Test of stability of noise
    # X_test_tor_noise = X_test_tor + \
    #     Variable(torch.randn(X_test_tor.size()) * 1.0)
    # final_net.eval()
    # test_pred = final_net(X_test_tor_noise)
    # loss_noise = accuracy(test_pred, y_test_tor)
    # print('\nAccuracy after adding noise: {0:.3f}%'.format(loss_noise))
