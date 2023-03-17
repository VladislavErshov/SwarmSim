import numpy as np
import matplotlib.pyplot as plt
from src.system import MultiAgentSystem


# Initialize agents

n_agents = 100
agent_dim = 2
goal_state = np.array([10, 0])

mas = MultiAgentSystem(n_agents, agent_dim, goal_state)
fstate = mas.get_full_state()
system_centroid = mas.get_cluster_centroids()[0]

# Initialize obstacles

# Simulation

n_steps = 100
val_each = 10
for sdx in range(n_steps):
    mas.update_system(step_size=0.1)
    fstate = mas.get_full_state()
    system_centroid = mas.get_cluster_centroids()[0]
    if sdx % val_each == 0:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
        ax.scatter(fstate[:, 0], fstate[:, 1], s=5, c='#0000ff99', marker='.')
        ax.scatter(goal_state[0], goal_state[1], s=10, c='r', marker='x')
        ax.scatter(system_centroid[0], system_centroid[1], s=40, c='#0000ff', marker='o')
        ax.set_xlim(-3, 11)
        ax.set_ylim(-5, 5)
        plt.show() 

