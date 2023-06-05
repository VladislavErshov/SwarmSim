import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.system import MultiAgentSystem, LinearAgentNd
import src.state_generator as gen
import warnings



warnings.filterwarnings('ignore')

# Initialize agents
n_agents = 1000
cluster_means = [(0, 5), (0, -5)]
cluster_std = 1
agent_dim = 2
control_dim = 2
goal_state = np.array([100, 0])
A = np.eye(agent_dim)
B = np.eye(agent_dim, control_dim)
mpc_n_t = 2
val_each = 1 # validate each val_each iteration

# Initialize obstacles [TODO]


# It won't work since the linear MPC is implemented, TODO FIX
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


def linear_mpc_distributred():
    n_steps = 10
    Q = np.eye(agent_dim)
    R = np.eye(control_dim)
    P = np.zeros((agent_dim, agent_dim))
    umin = None
    umax = None
    mas = MultiAgentSystem(n_agents, agent_dim, control_dim, goal_state, 
                           state_gen=gen.random_blobs, 
                           state_gen_args=[[A], [B], cluster_means, cluster_std, (-3, 3)])
    avg_goal_dist = mas.average_goal_distance
    cost_val = np.inf
    for sdx in range(n_steps):
        fstate = mas.get_full_state()
        cluster_centroids = mas.get_cluster_centroids()
        if sdx % val_each == 0:
            n_clusters = mas.n_clusters
            clust_labels = mas.clust_labels
            fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
            ax.scatter(goal_state[0], goal_state[1], s=30, c='k', marker='x')
            colors = cm.rainbow(np.linspace(0, 1, n_clusters))
            for cdx in range(n_clusters):
                agent_indices = np.where(clust_labels == cdx)[0]
                ax.scatter(fstate[agent_indices, 0], fstate[agent_indices, 1], 
                           s=5, c=colors[cdx], marker='.')
                ax.scatter(cluster_centroids[cdx][0], cluster_centroids[cdx][1], 
                           s=40, facecolors='none', edgecolors='#000000', marker='o')
            ax.set_title(f"Avg goal dist: {avg_goal_dist:.2f}; cost: {cost_val:.2f}")
            ax.set_xlim(-15, goal_state[0] * 1.2)
            ax.set_ylim(-10, 10)
            plt.show() 
        avg_goal_dist, cost_val = mas.update_system_mpc_distributed(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
    print(mas.control_solution_time)


def linear_mpc():
    n_steps = 10
    Q = np.eye(agent_dim * n_agents)
    R = np.eye(control_dim * n_agents)
    P = np.zeros((agent_dim * n_agents, agent_dim * n_agents))
    umin = None
    umax = None
    mas = MultiAgentSystem(n_agents, agent_dim, control_dim, goal_state, 
                           state_gen=gen.random_blobs, 
                           state_gen_args=[[A], [B], cluster_means, cluster_std, (-3, 3)])
    avg_goal_dist = mas.average_goal_distance
    cost_val = np.inf
    for sdx in range(n_steps):
        fstate = mas.get_full_state()
        cluster_centroids = mas.get_cluster_centroids()
        if sdx % val_each == 0:
            n_clusters = mas.n_clusters
            clust_labels = mas.clust_labels
            fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
            ax.scatter(goal_state[0], goal_state[1], s=30, c='k', marker='x')
            colors = cm.rainbow(np.linspace(0, 1, n_clusters))
            for cdx in range(n_clusters):
                agent_indices = np.where(clust_labels == cdx)[0]
                ax.scatter(fstate[agent_indices, 0], fstate[agent_indices, 1], 
                           s=5, c=colors[cdx], marker='.')
                ax.scatter(cluster_centroids[cdx][0], cluster_centroids[cdx][1], 
                           s=40, facecolors='none', edgecolors='#000000', marker='o')
            ax.set_title(f"Avg goal dist: {avg_goal_dist:.2f}; cost: {cost_val:.2f}")
            ax.set_xlim(-15, goal_state[0] * 1.2)
            ax.set_ylim(-10, 10)
            plt.show() 
        avg_goal_dist, cost_val = mas.update_system_mpc(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
    print(mas.control_solution_time)


if __name__ == '__main__':
    #simple_descent()
    linear_mpc()

