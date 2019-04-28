import torch
import torch.nn as nn
from collections import OrderedDict

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        # self.expand5x5 = nn.Conv2d(squeeze_planes, expand5x5_planes,
        #                            kernel_size=5, padding=2)
        # self.expand5x5_activation = nn.ReLU(inplace=True)        

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x)),
            # self.expand5x5_activation(self.expand5x5(x))
        ], 1)

class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 20, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()), 
            # ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire1', Fire(20, 20, 20, 20)),
            # ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire3', Fire(40, 50, 50, 50)),
            ('s3', nn.MaxPool2d(kernel_size=(4, 4), stride=4)),
            ('c3', nn.Conv2d(100, 10, kernel_size=(7, 7)))

        ]))

        self.loss = nn.Sequential(OrderedDict([
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        # print(output.shape)

        output = output.view(img.size(0), -1)
        # print(output.shape)
        output = self.loss(output)
        return output
