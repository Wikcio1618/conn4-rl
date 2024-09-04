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
        # self.conv1 = nn.Conv2d(1, 64, kernel_size = 4, padding='same') # OUT (64, 6, 7) 
        # self.conv2 = nn.Conv2d(64, 64, kernel_size = 4, padding='same') # OUT (64, 6, 7)

        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.width)
        self.fc4 = nn.Linear(self.width, self.width)



    def forward(self, x):
        # leaky because of https://towardsdatascience.com/deep-reinforcement-learning-and-monte-carlo-tree-search-with-connect-4-ba22a4713e7a
        # x = F.relu(self.conv1(x), inplace=True)
        # x = F.relu(self.conv2(x), inplace=True)
        # x = F.relu(self.conv3(x), inplace=True)
        # x = F.relu(self.conv4(x), inplace=True)
        # x = F.relu(self.conv5(x), inplace=True)
        # x = F.relu(self.conv6(x), inplace=True)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        x = F.relu(self.fc4(x), inplace=True)

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