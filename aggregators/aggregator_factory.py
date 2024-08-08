# Factory design pattern to be followed across the codebase

# Import necessary modules from decentai.
# The mean aggregator is imported and assigned the name 'mean_aggregator'.
from decentai.aggregators.mean import Aggregator as mean_aggregator

# The median aggregator is also imported, but with a different alias.
from decentai.aggregators.median import Aggregator as median_aggregator

# FedAvg aggregator is imported for further use.
from decentai.aggregators.median import Aggregator as fedavg_aggregator

# Importing the AggregatorInterface from decentai's aggregator_interface module.
# This interface serves as a common base class for all aggregators.
from decentai.aggregators.aggregator_interface import AggregatorInterface

def get_aggregator(aggregator_name: str) -> AggregatorInterface:
    """
    Determine which aggregator to return based on the provided configuration.
    
    Args:
        aggregator_name (str): The name of the aggregator, either 'mean', 'median' or 'fedavg'.

    Returns:
        AggregatorInterface: The corresponding aggregator interface object.

    Raises:
        ValueError: If the requested aggregator is not supported (i.e., neither 'mean', nor 'median', nor 'fedavg').
    """
    # Check if the aggregator_name is 'mean'.
    # The comparison is case-insensitive due to the use of the lowercase function.
    if aggregator_name.lower() == 'mean':
        return mean_aggregator  # Return the mean aggregator.
    
    # Check if the aggregator_name is 'median'.
    elif aggregator_name.lower() == 'median':
        return median_aggregator  # Return the median aggregator.
    
    # Check if the aggregator_name is 'fedavg'.
    elif aggregator_name.lower() == 'fedavg':
        return fedavg_aggregator  # Return the FedAvg aggregator.

    else:
        raise ValueError(f"Unsupported model: {aggregator_name}")  # Raise an error for unsupported models.
