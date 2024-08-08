# Import Abstract Base Classes (ABC) module from abc library.
# This module provides tools for writing abstract base classes in Python.

from abc import ABC, abstractmethod

# Define a class 'AggregatorInterface' that inherits from the abstract base class ABC.
# This is an abstract class defining an interface for decentralized learning aggregators.
class AggregatorInterface(ABC):
    """
    This is an abstract class defining an interface for decentralized learning aggregators.
    
    It provides one method:
        1. Aggregate: This method takes input list of agents (x) and applies the aggregation logic to it, returning the aggregated model.
        
    This methods is abstract, meaning it must be implemented by any concrete subclass that inherits from this interface.
    """
    
    # Define an abstract method 'Aggregate' that takes a list of agents and returns an aggregated model.
    # This is an abstract method. It should be implemented in any concrete subclass of AggregatorInterface.
    @abstractmethod
    def Aggregate(self, a, res):
        """
        This is an abstract method.
    
        Args:
            self: The instance of the class.
            x: A list of agents for model aggregation.
            res: A list of available resources perform model aggregation

        Returns:
            Aggregated global model.
        """
        pass