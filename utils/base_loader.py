from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List
import numpy as np
import torch

# This class represents a generic data loader for multiple agents.
class GenericDataLoader:
    # Constructor method. Initializes various attributes of the class instance.
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, num_agents: int, batch_size: int, shuffle: bool = True):
        self.train_dataset = train_dataset  # Holds the training dataset
        self.test_dataset = test_dataset  # Holds the testing dataset
        self.num_agents = num_agents  # Number of agents using this data loader
        self.batch_size = batch_size  # Size of each batch in the data loader
        self.shuffle = shuffle  # Flag indicating whether to randomly shuffle the dataset

    # Method to get the data loaders for training and testing.
    def get_loaders(self) -> Tuple[List[DataLoader], DataLoader]:
        # Split the training data among agents
        data_per_agent = len(self.train_dataset) // self.num_agents
        indices = list(range(len(self.train_dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        train_loaders = []  # List to hold the data loaders for each agent
        for i in range(self.num_agents):
            start_idx = i * data_per_agent
            end_idx = (i + 1) * data_per_agent
            agent_subset = Subset(self.train_dataset, indices[start_idx:end_idx])
            train_loader = DataLoader(agent_subset, batch_size=self.batch_size, shuffle=self.shuffle)
            train_loaders.append(train_loader)

        # Create the testing data loader
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loaders, test_loader

    # Method to get the agent indices (i.e., the start and end indices for each agent's dataset)
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

    # Method to get the dataset for a specific agent
    def get_agent_dataset(self, index: int) -> Subset:
        data_per_agent = len(self.train_dataset) // self.num_agents
        indices = list(range(len(self.train_dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        start_idx = index * data_per_agent
        end_idx = (index + 1) * data_per_agent
        return Subset(self.train_dataset, indices[start_idx:end_idx])

    # Method to get a batch of data from a given data loader
    def get_batch_from_loader(self, loader: DataLoader) -> torch.Tensor:
        batch_data = []
        for img, _ in loader:
            batch_data.append(img)
        return torch.stack(batch_data)

    # Method to get all the training data
    def get_all_train_data(self) -> List[torch.Tensor]:
        train_data = []
        for agent_loader in self.get_loaders()[0]:
            batch_data = self.get_batch_from_loader(agent_loader)
            train_data.extend(batch_data)
        return train_data

    # Method to get all the testing data
    def get_all_test_data(self) -> torch.Tensor:
        test_data = []
        for data, _ in self.get_loaders()[1]:
            test_data.append(data)
        return torch.stack(test_data)

# End of class definition
