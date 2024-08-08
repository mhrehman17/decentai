from decentai.coordinators.coordinator_interface import CoordinatorInterface

# This class coordinates the training process.
class TrainingCoordinator(CoordinatorInterface):
    # Constructor for the coordinator. It accepts agents and a resource manager.
    def __init__(self, agents, resource_manager):
        self.agents = agents  # List of agent objects
        self.resource_manager = resource_manager  # Resource manager object

    # This method is responsible for coordinating the training process among all agents.
    def coordinate(self, data_loaders):
        # Iterate over each agent and its corresponding data loader
        for agent, data_loader in zip(self.agents, data_loaders):
            # Find a resource suitable for training (e.g., a GPU)
            resource = self.resource_manager.find_resource({"task": "training"})
            print(f"Training {agent.agent_id} on {resource}")
            # Train the agent using the provided data loader
            agent.train(data_loader)