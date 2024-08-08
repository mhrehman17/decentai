# Import Abstract Base Classes (ABC) module from abc library.
from abc import ABC, abstractmethod

# Define a class 'AggregatorInterface' that inherits from the abstract base class ABC.
class AggregatorInterface(ABC):
    """
    This is an abstract class defining an interface for decentralized learning aggregators.
    
    It provides one method:
        1. Aggregator: This method takes input list of agents (x) and applies the aggregation logic to it, returning the aggregted model.
        
    This methods is abstract, meaning it must be implemented by any concrete subclass that inherits from this interface.
    """
    
    # Define an abstract method 'Aggregate' that takes list of agents and returns an aggregated model.
    @abstractmethod
    def Aggregate(self, a, res):
        """
        This is an abstract method. It should be implemented in any concrete subclass of AggregatorInterface.

        Args:
            self: The instance of the class.
            x: A list of agents for model aggregation.
            res: A list of available resources perform model aggregation
        Returns:
            Aggregated global model.
        """
        pass
    
    