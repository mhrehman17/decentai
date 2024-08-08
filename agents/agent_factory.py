# Import necessary modules from decentai library
# This line imports the MNIST agent class and the CIFAR-10 agent class from the decentai agents module.
from decentai.agents.mnistagent import Agent as MNISTAgent
from decentai.agents.cifar10agent import Agent as CIFAR10Agent

# Import the agent interface module 
# This module defines an interface for all agents in the system, providing a standard way of interacting with them.
from decentai.agents.agent_interface import AgentInterface

# Define a function to get an instance of the chosen agent
# This function takes a pipeline name as input and returns an instance of the corresponding agent based on that name.
def get_agent(pipeline_name: str) -> AgentInterface:
    """
    Get an instance of the specified agent based on pipeline name.
    
    Args:
        pipeline_name (str): Name of the pipeline, which determines the type of agent to return.

    Returns:
        AgentInterface: An instance of the chosen agent.
    
    Raises:
        ValueError: If the provided pipeline name is not supported.
    """

    # Check if the pipeline name matches 'mnist' and return the MNIST agent if so
    if pipeline_name.lower() == 'mnist':
        # This line returns an instance of the MNISTAgent class, which represents an agent for working with the MNIST dataset.
        return MNISTAgent

    # Check if the pipeline name matches 'cifar10' and return the CIFAR-10 agent if so
    elif pipeline_name.lower() == 'cifar10':
        # This line returns an instance of the CIFAR10Agent class, which represents an agent for working with the CIFAR-10 dataset.
        return CIFAR10Agent

    # If the pipeline name is neither 'mnist' nor 'cifar10', raise a ValueError
    else:
        # This line raises a ValueError with a message that includes the unsupported pipeline name.
        raise ValueError(f"Unsupported agent: {pipeline_name}")