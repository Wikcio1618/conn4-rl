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

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 7, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        
        x = self.fc3(x)
        return x

    #     # Convolutional layer
    #     self.conv1 = nn.Conv2d(1, self.conv_channels, kernel_size=4)
    #     # Flattened board representation after convolution
    #     conv_output_size = (self.height-3) * (self.width-3) * self.conv_channels
    #     # Fully connected layers
    #     self.fc1 = nn.Linear(conv_output_size, self.width)
    #     # Initialize weights as per He et. al.
    #     self.initialize_weights()

    # def forward(self, x):
    #     x = torch.relu(self.conv1(x))
    #     x = x.view(x.size(0), -1)  # Flatten the output of Conv2d
    #     x = torch.relu(self.fc1(x))
    #     return torch.softmax(x, dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def get_parameters(self):
        return [param.data.numpy().flatten() for param in self.parameters()]

    def set_parameters(self, parameters):
        start_idx = 0
        for param in self.parameters():
            # Calculate the number of elements in the parameter
            num_elements = param.numel()

            # Extract the corresponding slice from the 1D parameter array
            param_slice = parameters[start_idx:start_idx + num_elements]

            # Reshape and set the parameter
            param.data.copy_(torch.from_numpy(param_slice).reshape(param.size()))

            # Move to the next slice
            start_idx += num_elements

    def set_parameters_from_file(self, path):
        with open(path, 'r') as file:
            data = file.read()
            chromosome = np.array([float(element) for element in data.split(',') if element])
            self.set_parameters(chromosome)