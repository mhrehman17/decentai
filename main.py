# Import necessary modules from decentai system and its utilities
# This line imports modules needed for federated learning.
from decentai.system.federated_learning_system import FederatedLearningSystem

# This line imports a utility to load data.
from decentai.utils.loader_factory import get_data_loader

# This line imports a utility to create agents for the system.
from decentai.agents.agent_factory import get_agent

# This line imports a utility to create aggregators for the system.
from decentai.aggregators.aggregator_factory import get_aggregator

# This line imports PyTorch, a popular machine learning library.
import torch

# Define the main function for federated learning
# This is the main entry point for the program. It sets up and runs the federated learning experiment.
def main():
    """
    Main function for federated learning.
    """

    # Set parameters for the experiment
    # These variables set the number of agents, rounds, and batch size for the experiment.
    num_agents = 3  # Number of agents in the system
    num_rounds = 3  # Number of training rounds
    batch_size = 128  # Batch size for each agent's updates

    # Choose a pipeline name (e.g., 'mnist' or 'cifar10')
    # This variable specifies which dataset to use.
    pipeline_name = 'mnist'

    # Choose an aggregation strategy (e.g., 'mean', 'median', or 'fedavg')
    # This variable determines how the models from each agent are combined.
    aggregation_strategy = 'mean'

    # Set the device for computation
    # This line sets the device to use for computations, either a GPU if available, or CPU if not.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get data loaders
    # These lines load the dataset and split it into training and test sets.
    data_loader = get_data_loader(pipeline_name, num_agents, batch_size)
    train_loaders, test_loader = data_loader.get_loaders()
    
    # Choose an agent type based on the pipeline name
    # This line determines which type of agent to create for each device.
    agent_type = get_agent(pipeline_name)
    print("ML Pipeline Agent: "+str(agent_type))

    # Create a federated learning system
    # This line sets up and starts the federated learning experiment.
    print("Model Aggregation Strategy: "+str(aggregation_strategy))
    
    fl_system = FederatedLearningSystem(agent_type, num_agents, aggregation_strategy)
    fl_system.run(train_loaders, test_loader, num_rounds)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
