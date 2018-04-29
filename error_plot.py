import torch
import numpy as np
from aux import accuracy
import matplotlib.pyplot as plt
from torch.autograd import Variable


def error_plot(network, cv_data, X_test_tor, y_test_tor, iterations=100):
    # print('Fold {} of {}'.format(fold+1, K_FOLDS))
    K_FOLDS = 5
    vec_loss_noise = np.array([[]])
    k = 0
    for i in range(iterations):
        print("iteration {}".format(i))
        net = network
        net.train()
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
        net.eval()
        test_pred = net(X_test_tor)
        loss_test = accuracy(test_pred, y_test_tor)
        # print('\nAccuracy on test set: {0:.3f}%'.format(loss_test))

        # Test of stability of noise
        trail_noise = []
        for i in range(10):
            # print("i is {}".format(i))
            X_test_tor_noise = X_test_tor + \
                Variable(torch.randn(X_test_tor.size()) * 0.1)
            net.eval()
            test_pred = net(X_test_tor_noise)
            loss_noise = accuracy(test_pred, y_test_tor)
            trail_noise.append(loss_noise)
        trail_noise = np.array(trail_noise).reshape((-1, 1))
        # print(vec_loss_noise)
        trail_noise = trail_noise.reshape((1, -1))
        if k == 0:
            vec_loss_noise = trail_noise
        else:
            vec_loss_noise = np.concatenate((vec_loss_noise, trail_noise))
        k = k + 1
        print('\nAccuracy after adding noise: {0:.3f}%'.format(loss_noise))

    # Plot of error bar
    vec_loss_noise = np.array(vec_loss_noise)
    print(vec_loss_noise)
    mean = np.mean(vec_loss_noise, axis=1)
    deviation = np.std(vec_loss_noise, axis=1)
    ax = plt.figure().add_subplot(111)
    xaxis = np.array(range(1, iterations + 1))
    ax.errorbar(xaxis, mean, yerr=deviation, fmt='-.D', label='Error bar')
    plt.legend()
    plt.show()
