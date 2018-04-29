import torch
import torch.nn as nn


class Net(torch.nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = 1
        self.L2_PEN = 1e-6
        self.NUM_EPOCHS = 2000
        self.H2_SIZE = 64
        self.H3_SIZE = 64
        self.H1_DROPOUT = 0.9
        self.LR = 0.01
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
        self.NUM_EPOCHS = 500
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
        self.NUM_EPOCHS = 800
        self.H1_DROPOUT = 0.9
        self.LR = 0.01
        self.h1 = nn.Linear(self.INPUT_SIZE, self.H1_SIZE, bias=True)
        self.h2 = nn.Linear(self.H1_SIZE, self.H2_SIZE, bias=False)
        self.h3 = nn.Linear(self.H2_SIZE, self.H3_SIZE, bias=False)
        self.h4 = nn.Linear(self.H3_SIZE, self.OUTPUT_SIZE, bias=False)
        # self.h1 = nn.Sequential(nn.Linear(ResNet.INPUT_SIZE, ResNet.H1_SIZE, bias=True),
        #           nn.LeakyReLU(negative_slope=0.01),
        #           nn.Dropout(p=ResNet.H1_DROPOUT))
        # self.h2 = nn.Sequential(nn.Linear(TDRNN.H1_SIZE, TDRNN.H2_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        # self.h3 = nn.Sequential(nn.Linear(TDRNN.H2_SIZE, TDRNN.H3_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        # self.h4 = nn.Sequential(nn.Linear(TDRNN.H2_SIZE, TDRNN.H3_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        self.output = nn.Sequential(nn.Linear(self.H3_SIZE, self.OUTPUT_SIZE, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):
        LReLU = nn.LeakyReLU(negative_slope=0.01)
        # print(x.shape)
        # print(self.h1(x).shape)
        # print(self.h1.weight.t().shape)
        # print(torch.matmul(x, self.h1.weight.t()).shape)
        out = LReLU(-self.h1(x) + torch.matmul(x, self.h1.weight.t())) + x
        out = LReLU(-self.h2(out) + torch.matmul(out, self.h2.weight.t())) + out
        out = LReLU(-self.h3(out) + torch.matmul(out, self.h3.weight.t())) + out
        out = LReLU(-self.h4(out) + torch.matmul(out, self.h4.weight.t())) + out
        out = self.output(out)

        return out



class ODRNN(torch.nn.Module):

    # list of hyperparameters

    def __init__(self, input_size):
        super(ODRNN, self).__init__()
        self.INPUT_SIZE = input_size
        self.H1_SIZE = self.INPUT_SIZE
        self.H2_SIZE = self.INPUT_SIZE
        self.H3_SIZE = self.INPUT_SIZE
        self.H4_SIZE = self.INPUT_SIZE
        self.OUTPUT_SIZE = 1
        self.L2_PEN = 1e-6
        self.NUM_EPOCHS = 500
        self.H1_DROPOUT = 0.9
        self.LR = 0.1
        self.h1 = nn.Linear(int(self.INPUT_SIZE/2), int(self.H1_SIZE/2), bias=True)
        self.h2 = nn.Linear(int(self.H1_SIZE/2), int(self.H2_SIZE/2), bias=False)
        self.h3 = nn.Linear(int(self.H2_SIZE/2), int(self.H3_SIZE/2), bias=False)
        self.h4 = nn.Linear(int(self.H3_SIZE/2), int(self.H4_SIZE/2), bias=False)
        # self.h5 = nn.Linear(self.H4_SIZE, self.OUTPUT_SIZE, bias=False)
        # self.h1 = nn.Sequential(nn.Linear(ResNet.INPUT_SIZE, ResNet.H1_SIZE, bias=True),
        #           nn.LeakyReLU(negative_slope=0.01),
        #           nn.Dropout(p=ResNet.H1_DROPOUT))
        # self.h2 = nn.Sequential(nn.Linear(TDRNN.H1_SIZE, TDRNN.H2_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        # self.h3 = nn.Sequential(nn.Linear(TDRNN.H2_SIZE, TDRNN.H3_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        # self.h4 = nn.Sequential(nn.Linear(TDRNN.H2_SIZE, TDRNN.H3_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        self.output = nn.Sequential(nn.Linear(self.H4_SIZE, self.OUTPUT_SIZE, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):
        LReLU = nn.LeakyReLU(negative_slope=0.01)
        x_1 = x[:, 0:int(self.INPUT_SIZE/2)]
        x_2 = x[:, int(self.INPUT_SIZE/2):]
        # print(self.INPUT_SIZE)
        # print(x.shape)
        # print(self.h1(x).shape)
        # print(self.h1.weight.t().shape)
        # print(torch.matmul(x, self.h1.weight.t()).shape)
        # First layer
        x_1 = x_1 + LReLU(-torch.matmul(x_2, self.h1.weight.t()))
        x_2 = x_2 + LReLU(torch.matmul(x_1, self.h1.weight))
        x_1 = x_1 + LReLU(-torch.matmul(x_2, self.h2.weight.t()))
        x_2 = x_2 + LReLU(torch.matmul(x_1, self.h2.weight))
        x_1 = x_1 + LReLU(-torch.matmul(x_2, self.h3.weight.t()))
        x_2 = x_2 + LReLU(torch.matmul(x_1, self.h3.weight))
        x_1 = x_1 + LReLU(-torch.matmul(x_2, self.h4.weight.t()))
        x_2 = x_2 + LReLU(torch.matmul(x_1, self.h4.weight))
        # x_1 = x_1 + LReLU(-torch.matmul(x_2, self.h4.weight.t()))
        # x_2 = x_2 + LReLU(torch.matmul(x_1, self.h4.weight))
        out = torch.cat((x_1, x_2), dim=1)
        # out = LReLU(-self.h5(out) + torch.matmul(out, self.h4.weight.t())) + out
        out = self.output(out)

        return out



class SORNN(torch.nn.Module):

    # list of hyperparameters

    def __init__(self, input_size):
        super(SORNN, self).__init__()
        self.INPUT_SIZE = input_size
        self.H1_SIZE = self.INPUT_SIZE
        self.H2_SIZE = self.INPUT_SIZE
        self.H3_SIZE = self.INPUT_SIZE
        self.H4_SIZE = self.INPUT_SIZE
        self.OUTPUT_SIZE = 1
        self.L2_PEN = 1e-6
        self.NUM_EPOCHS = 100
        self.H1_DROPOUT = 0.9
        self.LR = 0.01
        self.h1 = nn.Linear(self.INPUT_SIZE, self.H1_SIZE, bias=True)
        self.h2 = nn.Linear(self.H1_SIZE, self.H2_SIZE, bias=False)
        self.h3 = nn.Linear(self.H2_SIZE, self.H3_SIZE, bias=False)
        self.h4 = nn.Linear(self.H3_SIZE, self.H4_SIZE, bias=False)
        self.h5 = nn.Linear(self.H4_SIZE, self.OUTPUT_SIZE, bias=False)
        self.drop = nn.Dropout(p=self.H1_DROPOUT)
        # self.h1 = nn.Sequential(nn.Linear(ResNet.INPUT_SIZE, ResNet.H1_SIZE, bias=True),
        #           nn.LeakyReLU(negative_slope=0.01),
        #           nn.Dropout(p=ResNet.H1_DROPOUT))
        # self.h2 = nn.Sequential(nn.Linear(TDRNN.H1_SIZE, TDRNN.H2_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        # self.h3 = nn.Sequential(nn.Linear(TDRNN.H2_SIZE, TDRNN.H3_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        # self.h4 = nn.Sequential(nn.Linear(TDRNN.H2_SIZE, TDRNN.H3_SIZE, bias=False),
        #                         nn.LeakyReLU(negative_slope=0.01))
        self.output = nn.Sequential(nn.Linear(self.H4_SIZE, self.OUTPUT_SIZE, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):
        LReLU = nn.LeakyReLU(negative_slope=0.01)
        # print(x.shape)
        # print(self.h1(x).shape)
        # print(self.h1.weight.t().shape)
        # print(torch.matmul(x, self.h1.weight.t()).shape)
        out_1 = self.drop(LReLU(-self.h1(x) + torch.matmul(x, self.h1.weight.t()))) + x
        out_2 = - torch.matmul(LReLU(self.h2(x)), self.h2.weight.t()) + 2*out_1 - x
        out_3 = - torch.matmul(LReLU(self.h3(x)), self.h3.weight.t()) + 2*out_2 - out_1
        out_4 = - torch.matmul(LReLU(self.h4(x)), self.h4.weight.t()) + 2*out_3 - out_2
        out = LReLU(-self.h5(out_4) + torch.matmul(out_4, self.h5.weight.t())) + out_4
        out = self.output(out)

        return out
