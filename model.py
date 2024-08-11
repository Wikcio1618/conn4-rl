import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConnectFourNN(nn.Module):
    def __init__(self):
        super(ConnectFourNN, self).__init__()

        # Board dimensions (6x7)
        self.height = 6
        self.width = 7

        # IN (1, 6, 7)
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 4, padding='same') # OUT (64, 6, 7) 
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 4, padding='same') # OUT (64, 6, 7)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size = 4, padding='same') # OUT (64, 6, 7)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size = 4, padding='same') # OUT (64, 6, 7)
        # self.conv5 = nn.Conv2d(64, 64, kernel_size = 4, padding='same') # OUT (64, 6, 7)
        # self.conv6 = nn.Conv2d(64, 64, kernel_size = 4, padding='same') # OUT (64, 6, 7)

        self.fc1 = nn.Linear(64*6*7, 64)
        self.fc2 = nn.Linear(64, 7)
        # self.fc3 = nn.Linear(64, 7)
        # self.fc4 = nn.Linear(7, self.width)


    def forward(self, x):
        # leaky because of https://towardsdatascience.com/deep-reinforcement-learning-and-monte-carlo-tree-search-with-connect-4-ba22a4713e7a
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3, inplace=True)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.3, inplace=True)
        # x = F.leaky_relu(self.conv3(x), negative_slope=0.3, inplace=True)
        # x = F.leaky_relu(self.conv4(x), negative_slope=0.3, inplace=True)
        # x = F.leaky_relu(self.conv5(x), negative_slope=0.3, inplace=True)
        # x = F.leaky_relu(self.conv6(x), negative_slope=0.3, inplace=True)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.leaky_relu(self.fc1(x), negative_slope=0.3, inplace=True)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.3, inplace=True)
        # x = F.leaky_relu(self.fc3(x), negative_slope=0.3, inplace=True)
        # x = F.leaky_relu(self.fc4(x), negative_slope=0.3, inplace=True)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)