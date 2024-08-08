import torch
from decentai.coordinators.coordinator_interface import CoordinatorInterface
from decentai.aggregators.aggregator_factory import get_aggregator
class AggregationCoordinator(CoordinatorInterface):
    """
    This class represents an aggregation coordinator.
    It coordinates multiple agents to aggregate their models.
    """

    def __init__(self, agents, resource_manager, selected_aggregator):
        """
        Initializes the coordinator with a list of agents and a resource manager.

        :param agents: A list of agents that need to be coordinated.
        :param resource_manager: An object responsible for managing resources.
        """
        self.agents = agents
        self.resource_manager = resource_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_aggregator = get_aggregator(selected_aggregator)  

    def coordinate(self):
        self.current_aggregator.Aggregate(self.agents, self.resource_manager, self.device)