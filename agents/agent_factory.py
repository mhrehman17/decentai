from decentai.agents.mnistagent import Agent as MNISTAgent
from decentai.agents.cifar10agent import Agent as CIFAR10Agent

from decentai.agents.agent_interface import AgentInterface

def get_agent(pipeline_name: str) -> AgentInterface:
    if pipeline_name.lower() == 'mnist':
        return MNISTAgent
    elif pipeline_name.lower() == 'cifar10':
        return CIFAR10Agent
    else:
        raise ValueError(f"Unsupported agent: {pipeline_name}")