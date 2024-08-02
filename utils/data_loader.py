# Import necessary libraries and modules from PyTorch's torchvision library
from torchvision import datasets, transforms

# Import torch module for tensor manipulation and data loading
import torch

# Define a type hint for return values of function get_data_loaders
from typing import Tuple, List

# Function to load MNIST dataset into multiple data loaders based on number of agents
def get_data_loaders(num_agents: int, batch_size: int) -> Tuple[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader]:
    """
    This function loads MNIST dataset and divides it equally among the specified number of agents.
    Each agent gets a portion of the data and uses it to train their respective models.
    The remaining data is used for testing.

    Args:
        num_agents (int): Number of agents or models being trained in parallel.
        batch_size (int): Batch size for each data loader.

    Returns:
        Tuple[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader]: A tuple containing a list of data loaders
        for training and a single data loader for testing.
    """

    # Define the transformation to apply to the dataset (e.g., convert to tensor, normalize pixel values)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST training dataset and download it if not present
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Load MNIST testing dataset without downloading if already present
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Calculate the number of samples per agent
    data_per_agent = len(full_dataset) // num_agents

    # Create a list to store data loaders for training
    train_loaders = [
        torch.utils.data.DataLoader(
            torch.utils.data.Subset(full_dataset, range(i * data_per_agent, (i + 1) * data_per_agent)),
            batch_size=batch_size, shuffle=True)
        for i in range(num_agents)
    ]
    
    # Create a data loader for testing
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Return the list of training data loaders and the testing data loader
    return train_loaders, test_loader