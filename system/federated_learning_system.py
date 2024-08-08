from decentai.managers.resource_manager import ResourceManager
from decentai.coordinators.training_coordinator import TrainingCoordinator
from decentai.coordinators.evaluation_coordinator import EvaluationCoordinator
from decentai.coordinators.aggregation_coordinator import AggregationCoordinator

# This class represents a federated learning system.
class FederatedLearningSystem:
    # Constructor for this class.
    def __init__(self, Agent, num_agents, aggregator):
        """
        Initializes the federated learning system.

        Args:
            Agent (object): The type of agent used in this system.
            num_agents (int): The number of agents in this system.
        """
        self.resource_manager = ResourceManager()
        # Creates a list of agents with names "Agent_<i>" where <i> is the index.
        self.agents = [Agent(f"Agent_{i}") for i in range(num_agents)]
        self.training_coordinator = TrainingCoordinator(self.agents, self.resource_manager)
        self.evaluation_coordinator = EvaluationCoordinator(self.agents, self.resource_manager)
        self.aggregation_coordinator = AggregationCoordinator(self.agents, self.resource_manager, aggregator)

    # Main method to run the federated learning system.
    def run(self, train_loaders, test_loader, num_rounds):
        """
        Runs the federated learning system for a specified number of rounds.

        Args:
            train_loaders (list): A list of data loaders for training.
            test_loader (object): A data loader for testing.
            num_rounds (int): The number of rounds to run the system.
        """
        # Runs the system for the specified number of rounds.
        for round in range(num_rounds):
            print(f"\nRound {round + 1}/{num_rounds}")
            self.training_coordinator.coordinate(train_loaders)
            avg_accuracy = self.evaluation_coordinator.coordinate(test_loader)
            print(f"Average accuracy: {avg_accuracy:.2f}%")
            self.aggregation_coordinator.coordinate()