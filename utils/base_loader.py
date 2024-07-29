from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List
import numpy as np

class GenericDataLoader:
    def __init__(self, dataset: Dataset, num_agents: int, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_loaders(self) -> Tuple[List[DataLoader], DataLoader]:
        data_per_agent = len(self.dataset) // self.num_agents
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        train_loaders = [
            DataLoader(
                Subset(self.dataset, indices[i * data_per_agent : (i + 1) * data_per_agent]),
                batch_size=self.batch_size,
                shuffle=self.shuffle
            )
            for i in range(self.num_agents)
        ]
        
        test_indices = indices[self.num_agents * data_per_agent:]
        test_loader = DataLoader(
            Subset(self.dataset, test_indices),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loaders, test_loader