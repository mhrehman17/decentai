from torchvision import datasets, transforms
from decentai.utils.base_loader import GenericDataLoader

class CIFAR10DataLoader(GenericDataLoader):
    def __init__(self, num_agents: int, batch_size: int, shuffle: bool = True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
        
        super().__init__(train_dataset, test_dataset, num_agents, batch_size, shuffle)