import numpy as np
from decentai.coordinators.coordinator_interface import CoordinatorInterface

class EvaluationCoordinator(CoordinatorInterface):
    def __init__(self, agents, resource_manager):
        self.agents = agents
        self.resource_manager = resource_manager

    def coordinate(self, test_loader):
        accuracies = []
        for agent in self.agents:
            resource = self.resource_manager.find_resource({"task": "evaluation"})
            print(f"Evaluating agent {agent.agent_id} on resource {resource}")
            accuracy = agent.evaluate(test_loader)
            accuracies.append(accuracy)
        return np.mean(accuracies)