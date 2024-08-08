# factory design pattern to be followed across the codebase

# Import necessary modules from decentai
from decentai.aggregators.mean import Aggregator as mean_aggregator  # Import mean aggregator
from decentai.aggregators.median import Aggregator as median_aggregator  # Import median aggregator
from decentai.aggregators.median import Aggregator as fedavg_aggregator  # Import FedAvg aggregator

# Import AggregatorInterface from decentai's aggregator_interface module
from decentai.aggregators.aggregator_interface import AggregatorInterface  # Interface for all aggregators

def get_aggregator(aggregator_name: str) -> AggregatorInterface:  # Define a function to get the aggregator
    """
    This function determines which aggregator to return based on the provided configuration.
    
    Args:
        aggregator_name (str): The name of the aggregator, either 'mean', 'median' or 'fedavg'.

    Returns:
        AggregatorInterface: The corresponding aggregator interface object.

    Raises:
        ValueError: If the requested aggregator is not supported (i.e., neither 'mean' nor 'median' or 'fedavg').
    """
    # Check if the aggregator_name is 'mean'
    if aggregator_name.lower() == 'mean':  # Convert to lowercase for case-insensitive comparison
        return mean_aggregator  # Return the mean aggregator

    # Check if the aggregator_name is 'median'
    elif aggregator_name.lower() == 'median':
        return median_aggregator  # Return the median aggregator
    
    # Check if the aggregator_name is 'fedavg'
    elif aggregator_name.lower() == 'fedavg':
        return fedavg_aggregator # Return the FedAvg aggregator
    
    else:
        raise ValueError(f"Unsupported model: {pipeline_name}")  # Raise an error for unsupported models
