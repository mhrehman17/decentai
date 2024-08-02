# Import Abstract Base Class (ABC) module from abc library
from abc import ABC, abstractmethod

"""
 Define a base class for an agent interface
 """
class AgentInterface(ABC):
    # Define an abstract method called train that takes data_loader as a parameter
    @abstractmethod
    def train(self, data_loader):
        """
        Abstract method for training the model
        """

    # Define an abstract method called evaluate that takes data_loader as a parameter
    @abstractmethod
    def evaluate(self, data_loader):
        """
        Abstract method for evaluating the model's performance
        """

    # Define an abstract method called get_model_params that does not take any parameters
    @abstractmethod
    def get_model_params(self):
        """
        Abstract method for retrieving the model's parameters
        """

    # Define an abstract method called set_model_params that takes a parameter named params
    @abstractmethod
    def set_model_params(self, params):
        """
        Abstract method for setting the model's parameters
        """
