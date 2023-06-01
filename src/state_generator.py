import numpy as np
import yaml



with open('cfg/seed.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    rnd_seed = data['RND_SEED']
    np.random.seed(rnd_seed)


def uniform_cube(agent_class, agent_dim, b=2, A=1, B=1):
    state = np.random.uniform(0, b, (agent_dim)) - 1
    agent = agent_class(A, B, agent_dim, state)
    return agent


def strict(agent_class, agent_dim, x0=[0,], A=1, B=1):
    assert agent_dim == np.array(x0).size
    state = np.array(x0)
    agent = agent_class(A, B, agent_dim, state)
    return agent

