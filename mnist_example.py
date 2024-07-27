from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from typing import List, Dict, Any
import numpy as np

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.model = Net()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.data = None

    def train(self, data_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Agent {} - Train Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    self.agent_id, batch_idx, len(data_loader), loss.item()))

    def evaluate(self, data_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)
        print('Agent {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.agent_id, test_loss, correct, len(data_loader.dataset), accuracy))
        return accuracy

    def get_model_params(self):
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_params(self, params):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = params[name].clone()

class ResourceManager:
    def __init__(self):
        self.available_resources = {"default": {"gpu": True}}

    def find_resource(self, requirements):
        return "default"  # For simplicity, always return the default resource

class TrainingCoordinator:
    def __init__(self, agents, resource_manager):
        self.agents = agents
        self.resource_manager = resource_manager

    def coordinate(self, data_loaders):
        for agent, data_loader in zip(self.agents, data_loaders):
            resource = self.resource_manager.find_resource({"task": "training"})
            print("Training agent {} on resource {}".format(agent.agent_id, resource))
            agent.train(data_loader)

class EvaluationCoordinator:
    def __init__(self, agents, resource_manager):
        self.agents = agents
        self.resource_manager = resource_manager

    def coordinate(self, test_loader):
        accuracies = []
        for agent in self.agents:
            resource = self.resource_manager.find_resource({"task": "evaluation"})
            print("Evaluating agent {} on resource {}".format(agent.agent_id, resource))
            accuracy = agent.evaluate(test_loader)
            accuracies.append(accuracy)
        return np.mean(accuracies)

class AggregationCoordinator:
    def __init__(self, agents, resource_manager):
        self.agents = agents
        self.resource_manager = resource_manager

    def coordinate(self):
        resource = self.resource_manager.find_resource({"task": "aggregation"})
        print("Aggregating models on resource {}".format(resource))
        
        # Implement FedAvg algorithm
        global_model = {}
        for name, _ in self.agents[0].get_model_params().items():
            global_model[name] = torch.stack([agent.get_model_params()[name] for agent in self.agents]).mean(0)
        
        # Update all agents with the new global model
        for agent in self.agents:
            agent.set_model_params(global_model)

class FederatedLearningSystem:
    def __init__(self, num_agents):
        self.agents = [Agent("Agent_{}".format(i)) for i in range(num_agents)]
        self.resource_manager = ResourceManager()
        self.training_coordinator = TrainingCoordinator(self.agents, self.resource_manager)
        self.evaluation_coordinator = EvaluationCoordinator(self.agents, self.resource_manager)
        self.aggregation_coordinator = AggregationCoordinator(self.agents, self.resource_manager)

    def run(self, train_loaders, test_loader, num_rounds):
        for round in range(num_rounds):
            print("\nRound {}/{}".format(round + 1, num_rounds))
            self.training_coordinator.coordinate(train_loaders)
            avg_accuracy = self.evaluation_coordinator.coordinate(test_loader)
            print("Average accuracy: {:.2f}%".format(avg_accuracy))
            self.aggregation_coordinator.coordinate()

# Data preparation
def get_data_loaders(num_agents):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # Load the entire dataset
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split the training data among agents
    data_per_agent = len(full_dataset) // num_agents
    train_loaders = [
        torch.utils.data.DataLoader(
            torch.utils.data.Subset(full_dataset, range(i * data_per_agent, (i + 1) * data_per_agent)),
            batch_size=64, shuffle=True)
        for i in range(num_agents)
    ]
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loaders, test_loader

# Main execution
if __name__ == "__main__":
    num_agents = 15
    num_rounds = 50
    
    train_loaders, test_loader = get_data_loaders(num_agents)
    fl_system = FederatedLearningSystem(num_agents)
    fl_system.run(train_loaders, test_loader, num_rounds)