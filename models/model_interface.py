# Import Abstract Base Classes (ABC) module from abc library.
from abc import ABC, abstractmethod

# Define a class 'ModelInterface' that inherits from the abstract base class ABC.
class ModelInterface(ABC):
    """
    This is an abstract class defining an interface for deep learning models.
    
    It provides two methods:
        1. forward: This method takes input data (x) and applies the model's prediction logic to it, returning the predicted output.
        2. to: This method changes the device the model operates on.
        
    Both of these methods are abstract, meaning they must be implemented by any concrete subclass that inherits from this interface.
    """
    
    # Define an abstract method 'forward' that takes input data (x) and returns a predicted output.
    @abstractmethod
    def forward(self, x):
        """
        This is an abstract method. It should be implemented in any concrete subclass of ModelInterface.

        Args:
            self: The instance of the class.
            x: Input data for prediction.
            
        Returns:
            Predicted output based on the model's logic.
        """
        pass
    
    # Define another abstract method 'to' that takes a device and changes the model's operation to it.
    @abstractmethod
    def to(self, device):
        """
        This is an abstract method. It should be implemented in any concrete subclass of ModelInterface.
        
        Args:
            self: The instance of the class.
            device: Device for the model's operations.
            
        Returns:
            None
        """
        pass
