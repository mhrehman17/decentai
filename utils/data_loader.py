import torch
from torchvision import datasets, transforms

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