import torch
from decentai.coordinators.coordinator_interface import CoordinatorInterface

class AggregationCoordinator(CoordinatorInterface):
    """
    This class represents an aggregation coordinator.
    It coordinates multiple agents to aggregate their models.
    """

    def __init__(self, agents, resource_manager):
        """
        Initializes the coordinator with a list of agents and a resource manager.

        :param agents: A list of agents that need to be coordinated.
        :param resource_manager: An object responsible for managing resources.
        """
        self.agents = agents
        self.resource_manager = resource_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def coordinate(self):
        """
        Coordinates the aggregation of models by multiple agents.

        :return: None
        """
        # Find a resource for the task
        resource = self.resource_manager.find_resource({"task": "aggregation"})
        print(f"Aggregating models on resource {resource}")

        global_model = {}
        # Aggregate model parameters from all agents
        for name, _ in self.agents[0].get_model_params().items():
            global_model[name] = torch.stack([agent.get_model_params()[name].to(self.device) for agent in self.agents]).mean(0)
        
        # Set the aggregated model parameters for each agent
        for agent in self.agents:
            agent.set_model_params({name: param.to(agent.device) for name, param in global_model.items()})