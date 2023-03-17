import numpy as np
import yaml



with open('cfg/seed.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    rnd_seed = data['RND_SEED']
    np.random.seed(rnd_seed)


def default_uniform_cube(agent_class, agent_dim, b=2):
    state = np.random.uniform(0, b, (agent_dim)) - 1
    agent = agent_class(agent_dim, state)
    return agent

