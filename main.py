# Import necessary modules and libraries
from decentai.system.federated_learning_system import FederatedLearningSystem  # Import Federated Learning System module
from decentai.utils.loader_factory import get_data_loader  # Import data loader factory
from decentai.agents.agent_factory import get_agent  # Import agent factory
from decentai.aggregators.aggregator_factory import get_aggregator  # Import aggregator factory
import torch  # Import PyTorch library
import time  # Import time module

# Define the main function for federated learning
def main():
    """
    Main function for federated learning.
    """

    # Set parameters for federated learning
    num_agents = 3  # Number of agents in the system
    num_rounds = 3  # Number of rounds for federated learning
    batch_size = 128  # Batch size for training

    # Specify the pipeline name and aggregation strategy
    pipeline_name = 'mnist'  # Name of the machine learning pipeline
    aggregation_strategy = 'mean'  # Aggregation strategy for model updates

    # Set the device (GPU or CPU) based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")  # Print the chosen device

    # Load data and create data loaders
    data_loader = get_data_loader(pipeline_name, num_agents, batch_size)
    train_loaders, test_loader = data_loader.get_loaders()
    
    # Get the agent type based on the pipeline name
    agent_type = get_agent(pipeline_name)
    print("ML Pipeline Agent:", str(agent_type))  # Print the agent type

    # Print the chosen aggregation strategy
    print("Model Aggregation Strategy:", str(aggregation_strategy))
    
    # Initialize and run the federated learning system
    fl_system = FederatedLearningSystem(agent_type, num_agents, aggregation_strategy)
    fl_system.run(train_loaders, test_loader, num_rounds)

# Run the main function when this script is executed as a standalone program
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
