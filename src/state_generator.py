import numpy as np
from sklearn.datasets import make_blobs



def _make_agents(agent_class, agent_dim, n_agents, As, Bs, states):
    agents = {}
    for idx in range(n_agents):
        if len(As) == n_agents:
            A = As[idx]
            B = Bs[idx]
        elif len(As) == 1:
            A = As[0]
            B = Bs[0]
        else:
            raise ValueError("The number of A's and B's should be equal to n_agents OR equal to 1")
        agents[idx] = agent_class(A, B, agent_dim, states[idx])
    return agents


def strict(agent_class, agent_dim=1, n_agents=1, 
           As=[1], Bs=[1], 
           x0s=[[0,]]):
    """
    Set initial agent states by state values provided in the arguments.
    
    Args:
        agent_class:        Class of an agent
        agent_dim:          Agent dimensionality
        n_agents:           Number of agents in the system to initialize
        As:                 State transition matrices for each agent
                            OR a list with a single matrix for all agents
        Bs:                 Control transition matrices for each agent
                            OR a list with a single matrix for all agents
        x0s:                Initial state values for all agents
    
    Returns:
        agents:             Dictionary of agent objects
    """
    assert n_agents == np.array(x0s).shape[0]
    assert agent_dim == np.array(x0s).shape[1]
    assert len(As) == len(Bs)
    states = np.array(x0s)
    agents = _make_agents(agent_class, agent_dim, n_agents, As, Bs, states)
    return agents


def random_blobs(agent_class, agent_dim=1, n_agents=1, 
                 As=[1], Bs=[1], 
                 cluster_data=1, cluster_std=1):
    """
    Set initial agent states by randomized truncated gaussian blobs
    parametrized by cluster centroids and diameters.
    
    Args:
        agent_class:        Class of an agent
        agent_dim:          Agent dimensionality
        n_agents:           Number of agents in the system to initialize
        As:                 State transition matrices for each agent
                            OR a list with a single matrix for all agents
        Bs:                 Control transition matrices for each agent
                            OR a list with a single matrix for all agents
        cluster_data:       [int]: number of clusters; centroid values randomized
                            [list of 'agent_dim'-tuples]: cluster centroid coordinates for each cluster
        cluster_std:        Standard deviation (spread) of agents within a cluster
    
    Returns:
        agents:             Dictionary of agent objects
    """
    assert len(As) == len(Bs)   
    states = make_blobs(n_samples=n_agents, n_features=agent_dim, 
                        centers=cluster_data, cluster_std=cluster_std)[0]
    agents = _make_agents(agent_class, agent_dim, n_agents, As, Bs, states)
    return agents

