from decentai.system.federated_learning_system import FederatedLearningSystem
from decentai.utils.loader_factory import get_data_loader
from decentai.agents.agent_factory import get_agent
import torch

def main():
    num_agents = 15
    num_rounds = 2
    batch_size = 64
    #pipeline_name = 'mnist'
    pipeline_name = 'cifar10'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_loader = get_data_loader(pipeline_name, num_agents, batch_size)
    train_loaders, test_loader = data_loader.get_loaders()
    
    agent_type = get_agent(pipeline_name)
    print(agent_type)
    fl_system = FederatedLearningSystem(agent_type, num_agents)
    fl_system.run(train_loaders, test_loader, num_rounds)

if __name__ == "__main__":
    main()