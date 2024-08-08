# Import necessary modules from decentai package
from decentai.managers.resource_manager import ResourceManager
from decentai.coordinators.training_coordinator import TrainingCoordinator
from decentai.coordinators.evaluation_coordinator import EvaluationCoordinator
from decentai.coordinators.aggregation_coordinator import AggregationCoordinator

# Define the FederatedLearningSystem class
class FederatedLearningSystem:
    # Initialize the system with an Agent, number of agents, and aggregator
    def __init__(self, Agent, num_agents, aggregator):
        # Create a ResourceManager instance to manage resources
        self.resource_manager = ResourceManager()
        
        # Create a list of Agent instances for each agent
        self.agents = [Agent(f"Agent_{i}") for i in range(num_agents)]
        # Initialize the training coordinator with agents and resource manager
        self.training_coordinator = TrainingCoordinator(self.agents, self.resource_manager)
        
        # Initialize the evaluation coordinator with agents and resource manager
        self.evaluation_coordinator = EvaluationCoordinator(self.agents, self.resource_manager)

        # Initialize the aggregation coordinator with agents, resource manager, and aggregator
        self.aggregation_coordinator = AggregationCoordinator(self.agents, self.resource_manager, aggregator)

    # Run the federated learning system for a specified number of rounds
    def run(self, train_loaders, test_loader, num_rounds):
        # Iterate over each round
        for round in range(num_rounds):
            print(f"\nRound {round + 1}/{num_rounds}")
            
            # Coordinate training with the training coordinator and provided loaders
            self.training_coordinator.coordinate(train_loaders)
            # Evaluate average accuracy using the evaluation coordinator and test loader
            avg_accuracy = self.evaluation_coordinator.coordinate(test_loader)
            print(f"Average accuracy: {avg_accuracy:.2f}%")
            
            # Coordinate aggregation with the aggregation coordinator
            self.aggregation_coordinator.coordinate()
