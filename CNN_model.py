import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN_Net, self).__init__()

        # Convolutional layer 1: input channels, 16 filters, 3x3 kernel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),  # Batch normalization for 16 feature maps
            nn.ReLU())

        # Convolutional layer 2: 16 filters, 3x3 kernel
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling with 2x2 window

        # Convolutional layer 3: 64 filters, 3x3 kernel
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),  # Batch normalization for 64 feature maps
            nn.ReLU())

        # Convolutional layer 4: 64 filters, 3x3 kernel
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU())

        # Convolutional layer 5: 64 filters, 3x3 kernel, with padding
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling with 2x2 window

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),  # Flattened output size, 128 neurons
            nn.ReLU(),
            nn.Linear(128, 128),  # 128 neurons
            nn.ReLU(),
            nn.Linear(128, num_classes))  # Output layer with number of classes

    def forward(self, x):
        # Forward pass through CNN layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Pass through fully connected layers
        return x
