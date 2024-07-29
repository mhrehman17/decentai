from abc import ABC, abstractmethod

class AgentInterface(ABC):
    @abstractmethod
    def train(self, data_loader):
        pass

    @abstractmethod
    def evaluate(self, data_loader):
        pass

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, params):
        pass