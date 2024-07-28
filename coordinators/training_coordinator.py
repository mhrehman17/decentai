class TrainingCoordinator:
    def __init__(self, agents, resource_manager):
        self.agents = agents
        self.resource_manager = resource_manager

    def coordinate(self, data_loaders):
        for agent, data_loader in zip(self.agents, data_loaders):
            resource = self.resource_manager.find_resource({"task": "training"})
            print(f"Training agent {agent.agent_id} on resource {resource}")
            agent.train(data_loader)