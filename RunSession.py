import numpy as np
import matplotlib.pyplot as plt
from src.system import MultiAgentSystem, LinearAgentNd
import src.state_generator as gen
import warnings



warnings.filterwarnings('ignore')

# Initialize agents
n_agents = 10
cluster_data = [(0, 20), (0, -20)]
agent_dim = 2
control_dim = 2
goal_state = np.array([1000, 0])
A = np.eye(agent_dim)
B = np.eye(agent_dim, control_dim)
val_each = 1 # validate each val_each iteration

# Initialize obstacles [TODO]


def simple_descent():
    n_steps = 100
    mas = MultiAgentSystem(n_agents, agent_dim, goal_state)
    fstate = mas.get_full_state()
    system_centroid = mas.get_cluster_centroids()[0]
    for sdx in range(n_steps):
        mas.update_system_simplified(step_size=1)
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


def linear_mpc():
    n_steps = 10
    Q = np.eye(agent_dim)
    R = np.eye(control_dim)
    P = np.zeros((agent_dim, agent_dim))
    mas = MultiAgentSystem(n_agents, agent_dim, control_dim, goal_state, 
                           state_gen=gen.uniform_cube, state_gen_args=[A, B, cluster_data, 1, (-10, 10)])
    for sdx in range(n_steps):
        fstate = mas.get_full_state()
        system_centroid = mas.get_cluster_centroids()[0]
        if sdx % val_each == 0:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
            ax.scatter(fstate[:, 0], fstate[:, 1], s=5, c='#0000ff99', marker='.')
            ax.scatter(goal_state[0], goal_state[1], s=10, c='r', marker='x')
            ax.scatter(system_centroid[0], system_centroid[1], s=40, facecolors='none', edgecolors='#0000ff', marker='o')
            ax.set_xlim(-3, 150)
            ax.set_ylim(-5, 5)
            plt.show() 
        mas.update_system_mpc(Q, R, P, n_t=2)


if __name__ == '__main__':
    #simple_descent()
    linear_mpc()

