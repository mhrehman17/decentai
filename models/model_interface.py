from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def to(self, device):
        pass