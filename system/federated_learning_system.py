from decentai.agents.agent import Agent
from decentai.managers.resource_manager import ResourceManager
from decentai.coordinators.training_coordinator import TrainingCoordinator
from decentai.coordinators.evaluation_coordinator import EvaluationCoordinator
from decentai.coordinators.aggregation_coordinator import AggregationCoordinator

class FederatedLearningSystem:
    def __init__(self, num_agents):
        self.agents = [Agent(f"Agent_{i}") for i in range(num_agents)]
        self.resource_manager = ResourceManager()
        self.training_coordinator = TrainingCoordinator(self.agents, self.resource_manager)
        self.evaluation_coordinator = EvaluationCoordinator(self.agents, self.resource_manager)
        self.aggregation_coordinator = AggregationCoordinator(self.agents, self.resource_manager)

    def run(self, train_loaders, test_loader, num_rounds):
        for round in range(num_rounds):
            print(f"\nRound {round + 1}/{num_rounds}")
            self.training_coordinator.coordinate(train_loaders)
            avg_accuracy = self.evaluation_coordinator.coordinate(test_loader)
            print(f"Average accuracy: {avg_accuracy:.2f}%")
            self.aggregation_coordinator.coordinate()