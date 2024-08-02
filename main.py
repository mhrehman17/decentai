# Import necessary modules from decentai system and its utilities
from decentai.system.federated_learning_system import FederatedLearningSystem  # <1>
from decentai.utils.loader_factory import get_data_loader  # <2>
from decentai.agents.agent_factory import get_agent  # <3>
import torch  # <4>

# Define the main function for federated learning
def main():  # <5>
    """
    Main function for federated learning.
    """

    # Set parameters for the experiment
    num_agents = 20  # Number of agents in the system  # <6>
    num_rounds = 200 # Number of training rounds  # <7>
    batch_size = 64  # Batch size for each agent's updates  # <8>

    # Choose a pipeline name (e.g., 'mnist' or 'cifar10')
    pipeline_name = 'mnist'  # <9>

    # Set the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <10>
    print(f"Using device: {device}")  # <11>

    # Get data loaders
    data_loader = get_data_loader(pipeline_name, num_agents, batch_size)  # <12>
    train_loaders, test_loader = data_loader.get_loaders()  # <13>
    
    # Choose an agent type based on the pipeline name
    agent_type = get_agent(pipeline_name)  # <14>
    print(agent_type)  # <15>

    # Create a federated learning system
    fl_system = FederatedLearningSystem(agent_type, num_agents)  # <16>
    fl_system.run(train_loaders, test_loader, num_rounds)  # <17>

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()  # <18>
