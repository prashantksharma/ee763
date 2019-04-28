import torch
import torch.nn as nn
from collections import OrderedDict

class Fire(nn.Module):

    def __init__(self, base_layer_dim,decision):
        super(Fire, self).__init__()

        P_expand_3x3 = 0.5
        squeeze_ratio = 0.25
        nExpand = int(base_layer_dim * decision)
        expand3x3_planes = int(nExpand * P_expand_3x3)
        expand1x1_planes = int(nExpand * (1 - P_expand_3x3))
        squeeze1x1_planes = int(squeeze_ratio * nExpand)

        self.squeeze = nn.Conv2d(base_layer_dim, squeeze1x1_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze1x1_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze1x1_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x)),
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
            ('c1', nn.Conv2d(1, 5, kernel_size=(3, 3))),
            ('relu1', nn.ReLU()),
            ('c3', nn.Conv2d(5, 10, kernel_size=(3, 3))),
            ('relu1', nn.ReLU()),  
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #10@14x14
            ('fire1', Fire(10,2)),
            ('fire3', Fire(20,1)),
            ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire4', Fire(20,2)),
            ('fire5', Fire(40,1)),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=1)),
            ('fire6', Fire(40, 0.5)),
            ('fire7', Fire(20, 1)),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c2', nn.Conv2d(20, 10, kernel_size=(3, 3)))

        ]))

        self.loss = nn.Sequential(OrderedDict([
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        #print(output.shape)

        output = output.view(img.size(0), -1)
        # print(output.shape)
        output = self.loss(output)
        return output
