import numpy as np
from decentai.coordinators.coordinator_interface import CoordinatorInterface

# Define a class that implements the CoordinatorInterface interface.
class EvaluationCoordinator(CoordinatorInterface):
    # Constructor to initialize the coordinator with agents and resource manager.
    def __init__(self, agents, resource_manager):
        self.agents = agents  # Initialize the list of agents.
        self.resource_manager = resource_manager  # Initialize the resource manager.

    # Method to coordinate the evaluation process for all agents.
    def coordinate(self, test_loader):
        accuracies = []  # Initialize a list to store the accuracy scores.
        for agent in self.agents:
            # Find a suitable resource for the current agent's evaluation task.
            resource = self.resource_manager.find_resource({"task": "evaluation"})
            print(f"Evaluating agent {agent.agent_id} on resource {resource}")
            # Evaluate each agent using the test loader and store the accuracy score.
            accuracy = agent.evaluate(test_loader)
            accuracies.append(accuracy)

        return np.mean(accuracies)  # Return the average accuracy across all agents.
