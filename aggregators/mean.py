# Import necessary modules
import torch
from decentai.aggregators.aggregator_interface import AggregatorInterface

# Define a distributed learning aggregator class
class Aggregator(AggregatorInterface):
    """This is a vanilla aggregator performing mean operations across all training agents."""
    # Initialize the model
    def __init__(self):
        """Initializes the model by calling the parent's constructor and defining aggregation attributes."""
        super(Aggregator, self).__init__()

    # Define the aggregation logic
    def aggregate(self, agents, resource_manager, device):
        """Coordinates the aggregation of models by multiple agents.

        :return: None
        """
        # Find a resource for the task
        resource = self.resource_manager.find_resource({"task": "aggregation"})
        print(f"Aggregating models on resource {resource}")

        global_model = {}
        # Aggregate model parameters from all agents
        for name, _ in agents[0].get_model_params().items():
            global_model[name] = torch.stack([agent.get_model_params()[name].to(device) for agent in agents]).mean(0)
        
        # Set the aggregated model parameters for each agent
        for agent in agents:
            agent.set_model_params({name: param.to(agent.device) for name, param in global_model.items()})