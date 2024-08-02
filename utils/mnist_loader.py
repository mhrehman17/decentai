from torchvision import datasets, transforms

# Import GenericDataLoader class from decentai.utils.base_loader module
from decentai.utils.base_loader import GenericDataLoader


class MNISTDataLoader(GenericDataLoader):

    # Initialize the MNISTDataLoader with num_agents, batch_size and shuffle as optional parameters
    def __init__(self, num_agents: int, batch_size: int, shuffle: bool = True):
        """
        This function initializes an instance of MNISTDataLoader.
        
        Parameters:
        num_agents (int): The number of agents in the dataset.
        batch_size (int): The size of each batch in the dataset.
        shuffle (bool): A boolean indicating whether to shuffle the data or not. Default value is True.
        """
        
        # Define a transformation for the MNIST dataset, consisting of ToTensor and Normalize operations
        transform = transforms.Compose([
            # Convert image to tensor
            transforms.ToTensor(),
            # Normalize pixel values to be between 0 and 1
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load the MNIST training dataset with specified transformation, download it if necessary
        train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
        
        # Load the MNIST testing dataset with specified transformation, download it if necessary
        test_dataset = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
        
        # Initialize the GenericDataLoader instance with training and testing datasets, num_agents, batch_size, and shuffle status
        super().__init__(train_dataset, test_dataset, num_agents, batch_size, shuffle)