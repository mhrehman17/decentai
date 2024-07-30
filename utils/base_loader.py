from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List
import numpy as np

class GenericDataLoader:
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, num_agents: int, batch_size: int, shuffle: bool = True):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_loaders(self) -> Tuple[List[DataLoader], DataLoader]:
        # Split training data among agents
        data_per_agent = len(self.train_dataset) // self.num_agents
        indices = list(range(len(self.train_dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        train_loaders = [
            DataLoader(
                Subset(self.train_dataset, indices[i * data_per_agent : (i + 1) * data_per_agent]),
                batch_size=self.batch_size,
                shuffle=self.shuffle
            )
            for i in range(self.num_agents)
        ]
        
        # Create test loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loaders, test_loader