import numpy as np
from sklearn.datasets import make_blobs
import yaml



with open('cfg/seed.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    rnd_seed = data['RND_SEED']
    np.random.seed(rnd_seed)


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


def strict(agent_class, agent_dim, n_agents, As=[1], Bs=[1], 
           x0s=[[0,]]):
    assert n_agents == np.array(x0s).shape[0]
    assert agent_dim == np.array(x0s).shape[1]
    assert len(As) == len(Bs)
    states = np.array(x0s)
    agents = _make_agents(agent_class, agent_dim, n_agents, As, Bs, states)
    return agents


def random_blobs(agent_class, agent_dim, n_agents, 
                 As=[1], Bs=[1], 
                 cluster_data=1, cluster_std=1, center_box=(-10, 10)):
    assert len(As) == len(Bs)   
    states = make_blobs(n_samples=n_agents, n_features=agent_dim, 
                        centers=cluster_data, cluster_std=cluster_std, center_box=center_box)[0]
    agents = _make_agents(agent_class, agent_dim, n_agents, As, Bs, states)
    return agents

