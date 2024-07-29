from decentai.utils.mnist_loader import MNISTDataLoader
from decentai.utils.cifar10_loader import CIFAR10DataLoader
from decentai.utils.base_loader import GenericDataLoader

def get_data_loader(dataset_name: str, num_agents: int, batch_size: int, shuffle: bool = True) -> GenericDataLoader:
    if dataset_name.lower() == 'mnist':
        return MNISTDataLoader(num_agents, batch_size, shuffle)
    elif dataset_name.lower() == 'cifar10':
        return CIFAR10DataLoader(num_agents, batch_size, shuffle)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")