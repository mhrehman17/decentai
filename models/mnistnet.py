# Import necessary modules
import torch
import torch.nn as nn
from decentai.models.model_interface import ModelInterface

# Define a neural network model class
class Net(nn.Module, ModelInterface):
    """
    This is a neural network model with two fully connected layers.
    """

    # Initialize the model
    def __init__(self):
        """
        Initializes the model by calling the parent's constructor and defining its layers.
        """
        super(Net, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(784, 64)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(64, 10)
        # Set the device for tensor computations (CPU in this case)
        self.device = torch.device("cpu")

    # Define the forward pass logic
    def forward(self, x):
        """
        This is where we define how our network processes inputs.
        """
        # Flatten the input tensor (assuming it's a batch of images with 784 features each)
        x = torch.flatten(x, 1)
        # Apply a ReLU activation function to the output of the first fully connected layer
        x = torch.relu(self.fc1(x))
        # Apply the second fully connected layer
        x = self.fc2(x)
        return x

    # Define a method for moving the model to a different device (GPU or CPU)
    def to(self, device):
        """
        This method is used to move the model to a different device.
        """
        super().to(device)
        # Update the internal device variable
        self.device = device
        return self