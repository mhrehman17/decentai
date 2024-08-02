# Import necessary modules from decentai
from decentai.models.mnistnet import Net as MNIST_Net  # Import MNIST network model
from decentai.models.cifarnet import Net as CIFAR10_Net  # Import CIFAR-10 network model

# Import ModelInterface from decentai's model_interface module
from decentai.models.model_interface import ModelInterface  # Interface for all models

def get_agent(pipeline_name: str) -> ModelInterface:  # Define a function to get the agent (model)
    """
    This function determines which model to return based on the provided pipeline name.
    
    Args:
        pipeline_name (str): The name of the pipeline, either 'mnist' or 'cifar10'.

    Returns:
        ModelInterface: The corresponding model interface object.

    Raises:
        ValueError: If the provided pipeline name is not supported (i.e., neither 'mnist' nor 'cifar10').
    """
    # Check if the pipeline_name is 'mnist'
    if pipeline_name.lower() == 'mnist':  # Convert to lowercase for case-insensitive comparison
        return MNIST_Net  # Return the MNIST network model

    # Check if the pipeline_name is 'cifar10'
    elif pipeline_name.lower() == 'cifar10':
        return CIFAR10_Net  # Return the CIFAR-10 network model
    else:
        raise ValueError(f"Unsupported model: {pipeline_name}")  # Raise an error for unsupported models
