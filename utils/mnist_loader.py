from torchvision import datasets, transforms
from decentai.utils.base_loader import GenericDataLoader

class MNISTDataLoader(GenericDataLoader):
    def __init__(self, num_agents: int, batch_size: int, shuffle: bool = True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        super().__init__(dataset, num_agents, batch_size, shuffle)