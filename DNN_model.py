import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, img_size):
        super(DNN, self).__init__()
        self.img_size = img_size  # Total number of input pixels

        # Fully connected layers
        self.fc1 = nn.Linear(img_size, 64 * 4 * 4)  # First hidden layer
        self.fc2 = nn.Linear(64 * 4 * 4, 128)  # Second hidden layer
        self.fc3 = nn.Linear(128, 2)  # Output layer with 2 classes

    def forward(self, x):
        x = x.view(-1, self.img_size)  # Flatten the input image
        x = F.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after second layer
        x = self.fc3(x)  # Output layer
        return x
