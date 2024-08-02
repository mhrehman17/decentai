import torch
import torch.nn as nn
import torch.optim as optim
from decentai.agents.agent_interface import AgentInterface
from decentai.models.mnistnet import Net

# Define a class for an agent, inheriting from the AgentInterface class
class Agent(AgentInterface):
    # Initialize the agent with its unique ID and device (GPU or CPU)
    def __init__(self, agent_id):
        self.agent_id = agent_id  # Store the agent's ID
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device (GPU or CPU) for computations
        self.model = Net().to(self.device)  # Initialize a neural network model and move it to the chosen device
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)  # Define an optimizer with learning rate of 0.01

    # Train the agent on the provided data loader
    def train(self, data_loader):
        self.model.train()  # Set the model to training mode
        for batch_idx, (data, target) in enumerate(data_loader):  # Iterate over batches in the data loader
            data, target = data.to(self.device), target.to(self.device)  # Move the data and targets to the chosen device
            self.optimizer.zero_grad()  # Zero the gradients of the model's parameters
            output = self.model(data)  # Pass the input data through the neural network model
            loss = nn.functional.cross_entropy(output, target)  # Calculate the cross-entropy loss between the predicted and actual outputs
            loss.backward()  # Backpropagate the gradients to update the model's parameters
            self.optimizer.step()  # Update the model's parameters using the optimizer
            if batch_idx % 10 == 0:  # Print the training progress every 10 batches
                print(f'Agent {self.agent_id} - Train Epoch: [{batch_idx}/{len(data_loader)}]\tLoss: {loss.item():.6f}')

    # Evaluate the agent on the provided data loader
    def evaluate(self, data_loader):
        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0  # Initialize the sum of the losses for this epoch
        correct = 0  # Initialize the number of correctly classified samples
        with torch.no_grad():  # Disable gradients for the evaluation process
            for data, target in data_loader:  # Iterate over batches in the data loader
                data, target = data.to(self.device), target.to(self.device)  # Move the data and targets to the chosen device
                output = self.model(data)  # Pass the input data through the neural network model
                test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()  # Calculate the cross-entropy loss between the predicted and actual outputs
                pred = output.argmax(dim=1, keepdim=True)  # Get the predicted class labels
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(data_loader.dataset)  # Calculate the average loss for this epoch
        accuracy = 100 * correct / len(data_loader.dataset)  # Calculate the classification accuracy
        print(f'Agent {self.agent_id} - Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)')
        return accuracy

    # Get a copy of the model's parameters
    def get_model_params(self):
        return {name: param.data.clone() for name, param in self.model.named_parameters()}  # Clone the model's parameters to create copies

    # Set the model's parameters using the provided parameter dictionary
    def set_model_params(self, params):
        with torch.no_grad():  # Disable gradients for the parameter updating process
            for name, param in self.model.named_parameters():
                param.data = params[name].clone()  # Update the model's parameters using the provided values
