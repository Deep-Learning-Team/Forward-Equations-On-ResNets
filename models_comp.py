import torch
import numpy as np
from aux import accuracy
import matplotlib.pyplot as plt
from torch.autograd import Variable


def models_comp(network_list, cv_data, X_test_tor, y_test_tor, name_network, iterations=100):
    # print('Fold {} of {}'.format(fold+1, K_FOLDS))
    ax1 = plt.figure().add_subplot(111)
    for i in range(len(network_list)):
        net = network_list[i]
        net.train()
        K_FOLDS = 5
        vec_loss = []
        for i in range(iterations):
            print("iteration {}".format(i))
            for fold in range(K_FOLDS):
                print("iteration {}".format(fold))
                X_train_tor, X_val_tor, y_train_tor, y_val_tor = cv_data[fold]
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

                    #   if VERBOSE and ((step % 100) == 0):
                    #       print('Step: {}'.format(step))
                    #       print('\tTrain Accuracy: {}'.format(hist_correct_train[step]))
                    #       print('\tVal Accuracy: {}'.format(hist_correct_val[step]))

                    # Update based on train performance
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()

                #   plt.plot(epochs, hist_correct_train, epochs, hist_correct_val)
                #   plt.title('Fold {} Accuracy'.format(fold+1))
                #   plt.xlabel('Epoch')
                #   plt.ylabel('Percent Accuracy')
                #   plt.ylim((0, 100))
                #   plt.legend(['Train', 'Validation'], loc='lower right')
                #   plt.show()

                # Accuracy of test set
                # print(X_test_tor.norm)
            # net.eval()
            # test_pred = net(X_test_tor)
            # loss_test = accuracy(test_pred, y_test_tor)
            # print('\nAccuracy on test set: {0:.3f}%'.format(loss_test))

            # Test of stability of noise
                # print("i is {}".format(i))
            X_test_tor_noise = X_test_tor + \
                Variable(torch.randn(X_test_tor.size()) * 0.1)
            net.eval()
            test_pred = net(X_test_tor_noise)
            loss_noise = accuracy(test_pred, y_test_tor)
            vec_loss.append(loss_noise)


        # Plots of stabilities of different models
        print(list(range(1, iterations+1)))
        print(vec_loss)
        ax1.plot(list(range(1, iterations+1)), vec_loss)
        # ax1.hold(True)
    ax1.legend(name_network)
    plt.ylim(50, 100)
    plt.show()
