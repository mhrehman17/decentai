import torch
import torch.nn as nn
import torch.optim as optim
from decentai.agents.agent_interface import AgentInterface
from decentai.models.mnistnet import Net

class Agent(AgentInterface):
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, data_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Agent {self.agent_id} - Train Epoch: [{batch_idx}/{len(data_loader)}]\tLoss: {loss.item():.6f}')

    def evaluate(self, data_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)
        print(f'Agent {self.agent_id} - Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)')
        return accuracy

    def get_model_params(self):
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_params(self, params):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = params[name].clone()