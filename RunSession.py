import numpy as np
import warnings
import yaml

from src.system import MultiAgentSystem
import src.state_generator as gen
from src.plot.plotter import plot_system



with open('cfg/seed.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    rnd_seed = data['RND_SEED']
    np.random.seed(rnd_seed)

warnings.filterwarnings('ignore')

# Initialize agents
n_agents = 1000 # number of agents
cluster_means = [(0, 5), (0, -5)] # coordinates of initial cluster centroids for each cluster
cluster_std = 0.5 # standard deviation for gaussian blob cluster initialization
clust_eps = 1 # epsilon-delta clustering parameter epsilon
agent_dim = 2 # dimensionality of each agent
control_dim = 2 # diemnsionality of control
goal_state = np.array([10, 0]) # goal point coordinates
A = np.eye(agent_dim) # initial matrix A (state transition) for a linear agent
B = np.eye(agent_dim, control_dim) # initial matrix B (control transition) for a linear agent
n_steps = 5 # number of MPC iterations
mpc_n_t = 15 # number of time steps per a single MPC iteration
mpc_n_t2 = 3 # number of time steps per a single MPC iteration for the micro-scale term
rad_max = 2. # target cluster radius maximum

# Initialize obstacles [TODO]


# It won't work since the linear MPC is implemented, TODO FIX
def simple_descent():
    for sdx in range(n_steps):
        pass


def linear_mpc():
    Q = np.eye(agent_dim)
    R = np.eye(control_dim)
    P = np.eye(agent_dim)#np.zeros((agent_dim, agent_dim))
    umin = None
    umax = None
    mas = MultiAgentSystem(n_agents, agent_dim, control_dim, goal_state, 
                           state_gen=gen.random_blobs, 
                           state_gen_args=[[A], [B], cluster_means, cluster_std],
                           clust_algo_params=[clust_eps, clust_eps])
    avg_goal_dist = mas.avg_goal_dist
    cost_val = np.inf
    for sdx in range(n_steps):
        plot_system(mas, goal_state, avg_goal_dist, cost_val)
        #avg_goal_dist, cost_val = mas.update_system_mpc_distributed(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
        #avg_goal_dist, cost_val = mas.update_system_mpc(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
        #avg_goal_dist, cost_val = mas.update_system_mpc_mesoonly(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
        avg_goal_dist, cost_val = mas.update_system_mpc_mesocoupling(Q, R, P, 
                                                                     n_t_mes=mpc_n_t, n_t_mic=mpc_n_t2, 
                                                                     rad_max=rad_max, 
                                                                     umax=umax, umin=umin)
    print("Total optimization time (s):", mas.control_solution_time)
    print("Final cost:", cost_val)
    print("Final average goal distance:", avg_goal_dist)


if __name__ == '__main__':
    #simple_descent()
    linear_mpc()

