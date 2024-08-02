# Import necessary modules from decentai
from decentai.utils.mnist_loader import MNISTDataLoader
from decentai.utils.cifar10_loader import CIFAR10DataLoader
from decentai.utils.base_loader import GenericDataLoader

# Define a function to get data loader based on the specified dataset name, number of agents, batch size, and shuffling option
def get_data_loader(dataset_name: str, num_agents: int, batch_size: int, shuffle: bool = True) -> GenericDataLoader:
    """
    This function returns a data loader instance based on the specified dataset name.
    
    Args:
        dataset_name (str): The name of the dataset to use. Can be 'mnist' or 'cifar10'.
        num_agents (int): The number of agents in the data loader.
        batch_size (int): The batch size for the data loader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        GenericDataLoader: An instance of the specified data loader class.
    
    Raises:
        ValueError: If the dataset name is not 'mnist' or 'cifar10'.
    """

    # Check if the dataset name matches 'mnist'
    if dataset_name.lower() == 'mnist':
        return MNISTDataLoader(num_agents, batch_size, shuffle)
    # Check if the dataset name matches 'cifar10'
    elif dataset_name.lower() == 'cifar10':
        return CIFAR10DataLoader(num_agents, batch_size, shuffle)
    
    # If none of the above conditions are met, raise a ValueError
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")