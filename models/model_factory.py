from decentai.models.mnistnet import Net as MNIST_Net
from decentai.models.cifarnet import Net as CIFAR10_Net

from decentai.models.model_interface import ModelInterface

def get_agent(pipeline_name: str) -> ModelInterface:
    if pipeline_name.lower() == 'mnist':
        return MNIST_Net
    elif pipeline_name.lower() == 'cifar10':
        return CIFAR10_Net
    else:
        raise ValueError(f"Unsupported model: {pipeline_name}")