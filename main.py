from decentai.system.federated_learning_system import FederatedLearningSystem
from decentai.utils.loader_factory import get_data_loader
import torch

def main():
    num_agents = 15
    num_rounds = 50
    batch_size = 64
    dataset_name = 'mnist'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_loader = get_data_loader(dataset_name, num_agents, batch_size)
    train_loaders, test_loader = data_loader.get_loaders()
    print(len(data_loader.dataset))

    fl_system = FederatedLearningSystem(num_agents)
    fl_system.run(train_loaders, test_loader, num_rounds)

if __name__ == "__main__":
    main()