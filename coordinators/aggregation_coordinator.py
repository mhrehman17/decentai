import torch
from decentai.coordinators.coordinator_interface import CoordinatorInterface

class AggregationCoordinator(CoordinatorInterface):
    def __init__(self, agents, resource_manager):
        self.agents = agents
        self.resource_manager = resource_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def coordinate(self):
        resource = self.resource_manager.find_resource({"task": "aggregation"})
        print(f"Aggregating models on resource {resource}")
        
        global_model = {}
        for name, _ in self.agents[0].get_model_params().items():
            global_model[name] = torch.stack([agent.get_model_params()[name].to(self.device) for agent in self.agents]).mean(0)
        
        for agent in self.agents:
            agent.set_model_params({name: param.to(agent.device) for name, param in global_model.items()})