# Import Abstract Base Class (ABC) module from abc library
# This import statement allows us to define abstract base classes and methods.
from abc import ABC, abstractmethod

""" Define a base class for an agent interface """
# The AgentInterface class is declared as an abstract base class (ABC).
class AgentInterface(ABC):
    # Abstract method definition: train.
    # This method trains the model using data from the provided data_loader.
    @abstractmethod
    def train(self, data_loader):
        """ 
        Abstract method for training the model. 
        """

    # Abstract method definition: evaluate.
    # This method evaluates the model's performance using data from the provided data_loader.
    @abstractmethod
    def evaluate(self, data_loader):
        """ 
        Abstract method for evaluating the model's performance. 
        """

    # Abstract method definition: get_model_params.
    # This method returns the current state of the model's parameters.
    @abstractmethod
    def get_model_params(self):
        """ 
        Abstract method for retrieving the model's parameters. 
        """

    # Abstract method definition: set_model_params.
    # This method sets new values to the model's parameters based on the provided params.
    @abstractmethod
    def set_model_params(self, params):
        """ 
        Abstract method for setting the model's parameters. 
        """
