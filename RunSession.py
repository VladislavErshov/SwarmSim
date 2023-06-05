import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings

from src.system import MultiAgentSystem, LinearAgentNd
import src.state_generator as gen
from src.plot.plotter import plot_system



warnings.filterwarnings('ignore')

# Initialize agents
n_agents = 1000 # number of agents
cluster_means = [(0, 5), (0, -5)] # coordinates of initial cluster centroids for each cluster
cluster_std = 1 # standard deviation for gaussian blob cluster initialization
cluster_diam = 2 # diameter of a cluster (equal for all clusters)
agent_dim = 2 # dimensionality of each agent
control_dim = 2 # diemnsionality of control
goal_state = np.array([100, 0]) # goal point coordinates
A = np.eye(agent_dim) # initial matrix A (state transition) for a linear agent
B = np.eye(agent_dim, control_dim) # initial matrix B (control transition) for a linear agent
n_steps = 10 # number of MPC iterations
mpc_n_t = 2 # number of time steps per a single MPC iteration

# Initialize obstacles [TODO]


# It won't work since the linear MPC is implemented, TODO FIX
def simple_descent():
    for sdx in range(n_steps):
        pass


def linear_mpc_distributred():
    Q = np.eye(agent_dim)
    R = np.eye(control_dim)
    P = np.zeros((agent_dim, agent_dim))
    umin = None
    umax = None
    mas = MultiAgentSystem(n_agents, agent_dim, control_dim, goal_state, 
                           state_gen=gen.random_blobs, 
                           state_gen_args=[[A], [B], cluster_means, cluster_std, cluster_diam])
    avg_goal_dist = mas.average_goal_distance
    cost_val = np.inf
    for sdx in range(n_steps):
        plot_system(mas, goal_state, avg_goal_dist, cost_val)
        avg_goal_dist, cost_val = mas.update_system_mpc_distributed(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
    print(mas.control_solution_time)


def linear_mpc():
    Q = np.eye(agent_dim * n_agents)
    R = np.eye(control_dim * n_agents)
    P = np.zeros((agent_dim * n_agents, agent_dim * n_agents))
    umin = None
    umax = None
    mas = MultiAgentSystem(n_agents, agent_dim, control_dim, goal_state, 
                           state_gen=gen.random_blobs, 
                           state_gen_args=[[A], [B], cluster_means, cluster_std, cluster_diam])
    avg_goal_dist = mas.average_goal_distance
    cost_val = np.inf
    for sdx in range(n_steps):
        plot_system(mas, goal_state, avg_goal_dist, cost_val)
        avg_goal_dist, cost_val = mas.update_system_mpc(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
    print(mas.control_solution_time)


if __name__ == '__main__':
    #simple_descent()
    linear_mpc()

