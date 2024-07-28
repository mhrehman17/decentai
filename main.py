from decentai.system.federated_learning_system import FederatedLearningSystem
from decentai.utils.data_loader import get_data_loaders

def main():
    num_agents = 15
    num_rounds = 50

    train_loaders, test_loader = get_data_loaders(num_agents)
    fl_system = FederatedLearningSystem(num_agents)
    fl_system.run(train_loaders, test_loader, num_rounds)

if __name__ == "__main__":
    main()