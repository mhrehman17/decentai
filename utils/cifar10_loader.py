# Import necessary libraries from torchvision.
from torchvision import datasets, transforms

# Import GenericDataLoader class from decentai.utils.base_loader.
from decentai.utils.base_loader import GenericDataLoader

# Define a class called CIFAR10DataLoader that inherits from GenericDataLoader.
class CIFAR10DataLoader(GenericDataLoader):
    # Initialize the class with parameters num_agents, batch_size, and shuffle (default is True).
    def __init__(self, num_agents: int, batch_size: int, shuffle: bool = True):
        # Define a transform composition that includes ToTensor and Normalize.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load the CIFAR10 dataset for training with True and download it if necessary.
        train_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
        
        # Load the CIFAR10 dataset for testing with True and download it if necessary. Note that this line is likely incorrect, as test data should not be downloaded multiple times.
        test_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
        
        # Call the superclass's constructor to initialize the GenericDataLoader.
        super().__init__(train_dataset, test_dataset, num_agents, batch_size, shuffle)

