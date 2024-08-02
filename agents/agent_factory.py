# Import necessary modules from decentai library
from decentai.agents.mnistagent import Agent as MNISTAgent
from decentai.agents.cifar10agent import Agent as CIFAR10Agent

# Import agent interface module
from decentai.agents.agent_interface import AgentInterface

# Define a function to get an instance of the chosen agent
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
    if pipeline_name.lower() == 'mnist':
        # Return MNIST agent if pipeline name is 'mnist'
        return MNISTAgent
    elif pipeline_name.lower() == 'cifar10':
        # Return CIFAR-10 agent if pipeline name is 'cifar10'
        return CIFAR10Agent
    else:
        # Raise ValueError if the provided pipeline name is not supported
        raise ValueError(f"Unsupported agent: {pipeline_name}")