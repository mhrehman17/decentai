from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List
import numpy as np
import torch

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

        train_loaders = []
        for i in range(self.num_agents):
            start_idx = i * data_per_agent
            end_idx = (i + 1) * data_per_agent
            agent_subset = Subset(self.train_dataset, indices[start_idx:end_idx])
            train_loader = DataLoader(agent_subset, batch_size=self.batch_size, shuffle=self.shuffle)
            train_loaders.append(train_loader)

        # Create test loader
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loaders, test_loader

    def get_agent_indices(self) -> List[int]:
        data_per_agent = len(self.train_dataset) // self.num_agents
        indices = list(range(len(self.train_dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        agent_indices = []
        for i in range(self.num_agents):
            start_idx = i * data_per_agent
            end_idx = (i + 1) * data_per_agent
            agent_indices.append(list(range(start_idx, end_idx)))
        return agent_indices

    def get_agent_dataset(self, index: int) -> Subset:
        data_per_agent = len(self.train_dataset) // self.num_agents
        indices = list(range(len(self.train_dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        start_idx = index * data_per_agent
        end_idx = (index + 1) * data_per_agent
        return Subset(self.train_dataset, indices[start_idx:end_idx])

    def get_batch_from_loader(self, loader: DataLoader) -> torch.Tensor:
        batch_data = []
        for img, _ in loader:
            batch_data.append(img)
        return torch.stack(batch_data)

    def get_all_train_data(self) -> List[torch.Tensor]:
        train_data = []
        for agent_loader in self.get_loaders()[0]:
            batch_data = self.get_batch_from_loader(agent_loader)
            train_data.extend(batch_data)
        return train_data

    def get_all_test_data(self) -> torch.Tensor:
        test_data = []
        for data, _ in self.get_loaders()[1]:
            test_data.append(data)
        return torch.stack(test_data)

 
