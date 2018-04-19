import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = 1
        self.L2_PEN = 1e-6
        self.NUM_EPOCHS = 200
        self.H2_SIZE = 64
        self.H3_SIZE = 64
        self.H1_DROPOUT = 0.9
        self.LR = 0.1
        self.H1_SIZE = self.INPUT_SIZE
        self.h1 = nn.Sequential(nn.Linear(self.INPUT_SIZE, self.H1_SIZE, bias=True),
                                nn.LeakyReLU(negative_slope=0.01),
                                nn.Dropout(p=self.H1_DROPOUT))
        self.h2 = nn.Sequential(nn.Linear(self.H1_SIZE, self.H2_SIZE, bias=False),
                                nn.LeakyReLU(negative_slope=0.01))
        self.h3 = nn.Sequential(nn.Linear(self.H2_SIZE, self.H3_SIZE, bias=False),
                                nn.LeakyReLU(negative_slope=0.01))
        self.output = nn.Sequential(nn.Linear(self.H3_SIZE, self.OUTPUT_SIZE, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):

        out = self.h1(x)
        out = self.h2(out)
        out = self.h3(out)
        out = self.output(out)

        return out


class ResNet(torch.nn.Module):


    def __init__(self, input_size):
        super(ResNet, self).__init__()
        self.INPUT_SIZE = input_size
        self.H1_SIZE = self.INPUT_SIZE
        self.H2_SIZE = self.INPUT_SIZE
        self.H3_SIZE = self.INPUT_SIZE
        self.OUTPUT_SIZE = 1
        self.L2_PEN = 1e-6
        self.NUM_EPOCHS = 200
        self.H1_DROPOUT = 0.9
        self.LR = 0.1
        self.h1 = nn.Sequential(nn.Linear(self.INPUT_SIZE, self.H1_SIZE, bias=True),
                                nn.LeakyReLU(negative_slope=0.01),
                                nn.Dropout(p=self.H1_DROPOUT))
        self.h2 = nn.Sequential(nn.Linear(self.H1_SIZE, self.H2_SIZE, bias=False),
                                nn.LeakyReLU(negative_slope=0.01))
        self.h3 = nn.Sequential(nn.Linear(self.H2_SIZE, self.H3_SIZE, bias=False),
                                nn.LeakyReLU(negative_slope=0.01))
        self.h4 = nn.Sequential(nn.Linear(self.H2_SIZE, self.H3_SIZE, bias=False),
                                nn.LeakyReLU(negative_slope=0.01))
        self.output = nn.Sequential(nn.Linear(self.H3_SIZE, self.OUTPUT_SIZE, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):

        out = self.h1(x) + x
        out = self.h2(out) + out
        out = self.h3(out) + out
        out = self.h4(out) + out
        out = self.output(out)

        return out


class TDRNN(torch.nn.Module):

    # list of hyperparameters

    def __init__(self, input_size):
        super(TDRNN, self).__init__()
        self.INPUT_SIZE = input_size
        self.H1_SIZE = self.INPUT_SIZE
        self.H2_SIZE = self.INPUT_SIZE
        self.H3_SIZE = self.INPUT_SIZE
        self.OUTPUT_SIZE = 1
        self.L2_PEN = 1e-6
        self.NUM_EPOCHS = 200
        self.H1_DROPOUT = 0.9
        self.LR = 0.1
        self.h1 = nn.Linear(self.INPUT_SIZE, self.H1_SIZE, bias=True)
        self.h2 = nn.Linear(self.H1_SIZE, self.H2_SIZE, bias=True)
        self.h3 = nn.Linear(self.H2_SIZE, self.H3_SIZE, bias=True)
        self.h4 = nn.Linear(self.H3_SIZE, self.OUTPUT_SIZE, bias=True)
        # self.h1 = nn.Sequential(nn.Linear(ResNet.INPUT_SIZE, ResNet.H1_SIZE, bias=True),
        #           nn.LeakyReLU(negative_slope=0.01),
        #           nn.Dropout(p=ResNet.H1_DROPOUT))
#         self.h2 = nn.Sequential(nn.Linear(TDRNN.H1_SIZE, TDRNN.H2_SIZE, bias=False),
#                                 nn.LeakyReLU(negative_slope=0.01))
#         self.h3 = nn.Sequential(nn.Linear(TDRNN.H2_SIZE, TDRNN.H3_SIZE, bias=False),
#                                 nn.LeakyReLU(negative_slope=0.01))
#         self.h4 = nn.Sequential(nn.Linear(TDRNN.H2_SIZE, TDRNN.H3_SIZE, bias=False),
#                                 nn.LeakyReLU(negative_slope=0.01))
        self.output = nn.Sequential(nn.Linear(self.H3_SIZE, self.OUTPUT_SIZE, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):
        LReLU = nn.LeakyReLU(negative_slope=0.01)
        # print(x.shape)
        # print(self.h1(x).shape)
        # print(self.h1.weight.t().shape)
        # print(torch.matmul(x, self.h1.weight.t()).shape)
        out = LReLU(self.h1(x) - torch.matmul(x, self.h1.weight.t())) + x
        out = LReLU(self.h2(out) - torch.matmul(out, self.h2.weight.t())) + out
        out = LReLU(self.h3(out) - torch.matmul(out, self.h3.weight.t())) + out
        out = LReLU(self.h4(out) - torch.matmul(out, self.h4.weight.t())) + out
        out = self.output(out)

        return out
