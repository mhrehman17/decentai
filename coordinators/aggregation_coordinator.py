import torch

class AggregationCoordinator:
    def __init__(self, agents, resource_manager):
        self.agents = agents
        self.resource_manager = resource_manager

    def coordinate(self):
        resource = self.resource_manager.find_resource({"task": "aggregation"})
        print(f"Aggregating models on resource {resource}")
        
        # Implement FedAvg algorithm
        global_model = {}
        for name, _ in self.agents[0].get_model_params().items():
            global_model[name] = torch.stack([agent.get_model_params()[name] for agent in self.agents]).mean(0)
        
        # Update all agents with the new global model
        for agent in self.agents:
            agent.set_model_params(global_model)