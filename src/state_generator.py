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


def strict(
    agent_class,
    agent_dim=1,
    n_agents=1,
    As=[1],
    Bs=[1],
    x0s=[[0, ]]
):
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

    return _make_agents(agent_class, agent_dim, n_agents, As, Bs, states)


def random_blobs(
    agent_class,
    agent_dim=1,
    n_agents=1,
    As=[1],
    Bs=[1],
    cluster_data=1,
    cluster_std=1
):
    """
    Set initial agent states by randomized gaussian blobs
    parametrized by cluster centroids and standard deviation.
    
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

    states = make_blobs(
        n_samples=n_agents,
        n_features=agent_dim,
        centers=cluster_data,
        cluster_std=cluster_std
    )[0]

    return _make_agents(agent_class, agent_dim, n_agents, As, Bs, states)


def uni_ball(
    agent_class,
    agent_dim=1,
    n_agents=1,
    As=[1],
    Bs=[1],
    cluster_data=[(0, 0)],
    cluster_rad=1
):
    """
    Set initial agent states by randomized uniform blobs 
    parametrized by cluster centroids and a radius.
    
    Args:
        agent_class:        Class of an agent
        agent_dim:          Agent dimensionality
        n_agents:           Number of agents in the system to initialize
        As:                 State transition matrices for each agent
                            OR a list with a single matrix for all agents
        Bs:                 Control transition matrices for each agent
                            OR a list with a single matrix for all agents
        cluster_data:       Cluster centroid coordinates for each cluster
        cluster_rad:        Radius of a cluster
    
    Returns:
        agents:             Dictionary of agent objects
    """
    assert len(As) == len(Bs)
    n_ag_per_cluster = n_agents // len(cluster_data)
    states = np.zeros((n_agents, agent_dim))
    for cdx, centroid in enumerate(cluster_data):
        r = np.sqrt(np.random.uniform(0, cluster_rad, n_ag_per_cluster))
        a = np.pi * np.random.uniform(0, 2, n_ag_per_cluster)
        xy = np.array([r * np.cos(a), r * np.sin(a)]).T
        xy += centroid
        states[cdx*n_ag_per_cluster:(cdx+1)*n_ag_per_cluster, :] = xy

    return _make_agents(agent_class, agent_dim, n_agents, As, Bs, states)
