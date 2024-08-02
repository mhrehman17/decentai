# Import necessary modules from PyTorch.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import ModelInterface from decentai.models.model_interface.
from decentai.models.model_interface import ModelInterface

# Define a neural network class that inherits from both nn.Module and ModelInterface.
class Net(nn.Module, ModelInterface):
    # Initialize the neural network with its layers.
    def __init__(self):
        # Call the parent constructor to initialize the module.
        super().__init__()
        
        # Define the first convolutional layer with 3 input channels, 6 output channels, and a kernel size of 5.
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        # Define the maximum pooling layer with a kernel size of 2x2.
        self.pool = nn.MaxPool2d(2, 2)
        
        # Define the second convolutional layer with 6 input channels, 16 output channels, and a kernel size of 5.
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Define the first fully connected (dense) layer with an input dimension of 16*5*5 and an output dimension of 120.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # Define the second fully connected (dense) layer with an input dimension of 120 and an output dimension of 84.
        self.fc2 = nn.Linear(120, 84)
        
        # Define the third fully connected (dense) layer with an input dimension of 84 and an output dimension of 10.
        self.fc3 = nn.Linear(84, 10)

    # Define the forward pass method for the neural network.
    def forward(self, x):
        # Apply the first convolutional layer to the input, followed by a ReLU activation function and maximum pooling.
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply the second convolutional layer to the output of the previous step, followed by a ReLU activation function and maximum pooling.
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten all dimensions except for the batch dimension.
        x = torch.flatten(x, 1)
        
        # Apply the first fully connected (dense) layer to the flattened output, followed by a ReLU activation function.
        x = F.relu(self.fc1(x))
        
        # Apply the second fully connected (dense) layer to the output of the previous step, followed by a ReLU activation function.
        x = F.relu(self.fc2(x))
        
        # Apply the third fully connected (dense) layer to the output of the previous step.
        x = self.fc3(x)
        
        # Return the output of the forward pass.
        return x

    # Define a method for moving the neural network to a specified device (e.g., GPU).
    def to(self, device):
        # Move the parent module to the specified device.
        super().to(device)
        
        # Store the device in an instance variable for later use.
        self.device = device
        
        # Return the neural network.
        return self
