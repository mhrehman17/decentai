import torch
from decentai.coordinators.coordinator_interface import CoordinatorInterface
from decentai.aggregators.aggregator_factory import get_aggregator

# Define a new class that inherits from CoordinatorInterface
class AggregationCoordinator(CoordinatorInterface):
    """
    This is an implementation of the CoordinatorInterface.
    It coordinates multiple agents and resource manager to achieve a common goal.

    Attributes:
        agents: A list of agents that need to be coordinated.
        resource_manager: An object responsible for managing resources.
        device: The device (GPU or CPU) where computations will take place.
        current_aggregator: An aggregator selected based on the provided aggregator name.
    """

    # Define the constructor, which sets up the necessary attributes.
    def __init__(self, agents, resource_manager, selected_aggregator):
        """
        Initializes an instance of AggregationCoordinator.

        Args:
            agents (list): A list of agents that need to be coordinated.
            resource_manager: An object responsible for managing resources.
        """ 
        self.agents = agents
        self.resource_manager = resource_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_aggregator = get_aggregator(selected_aggregator)  

    def coordinate(self):
        self.current_aggregator.Aggregate(self.agents, self.resource_manager, self.device)