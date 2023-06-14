import numpy as np
import warnings
import yaml
import argparse
import itertools
import os
import pandas as pd

from src.system import MultiAgentSystem
import src.state_generator as gen
import src.plot.plotter as pltr
from src.multiprocessing.mp_wrapper import mp_kwargs_wrapper



with open('cfg/seed.yaml') as f:
    seed_data = yaml.load(f, Loader=yaml.FullLoader)
    rnd_seed = seed_data['RND_SEED']
    #np.random.seed(rnd_seed)

warnings.filterwarnings('ignore')

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help="Config name", required=False, default='exprt_1')
args = parser.parse_args()

# Initialize obstacles [TODO]

def linear_mpc(
        n_agents = 1000, # number of agents
        cluster_means = [(0, 5), (0, -5)], # coordinates of initial cluster centroids for each cluster
        cluster_std = 1, # standard deviation for gaussian blob cluster initialization
        clust_eps = 1.5, # epsilon-delta clustering parameter epsilon
        agent_dim = 2, # dimensionality of each agent
        control_dim = 2, # diemnsionality of control
        goal_state = np.array([10, 0]), # goal point coordinates
        A = np.eye(2), # initial matrix A (state transition) for a linear agent
        B = np.eye(2, 2), # initial matrix B (control transition) for a linear agent
        u_bound = 2, # control constraint absolute value
        n_steps = 10, # number of MPC iterations
        mpc_n_t = 16, # number of time steps per a single MPC iteration
        mpc_n_t2 = 2, # number of time steps per a single MPC iteration for the micro-scale term
        rad_max = 2., # target cluster radius maximum
        lap_lambda = 1., # coupling weight
        control_strategy = 'mesocoup', # control strategy
        plot_dynamics = False, # 'True' if draw system dynamics step-by-step
    ):
    Q = np.eye(agent_dim)
    R = np.eye(control_dim)
    P = np.eye(agent_dim)#np.zeros((agent_dim, agent_dim))
    umax = u_bound
    umin = -u_bound
    mas = MultiAgentSystem(n_agents, agent_dim, control_dim, goal_state, 
                           state_gen=gen.random_blobs, 
                           state_gen_args=[[A], [B], cluster_means, cluster_std],
                           clust_algo_params=[clust_eps, clust_eps])
    avg_goal_dist = mas.avg_goal_dist
    cost_val = np.inf
    for sdx in range(n_steps):
        if plot_dynamics:
            pltr.system_state(mas, goal_state, avg_goal_dist, cost_val)
        if control_strategy == 'micro':
            avg_goal_dist, cost_val = mas.update_system_mpc(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
        elif control_strategy == 'microdist':
            avg_goal_dist, cost_val = mas.update_system_mpc_distributed(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
        elif control_strategy == 'meso':
            avg_goal_dist, cost_val = mas.update_system_mpc_mesoonly(Q, R, P, n_t=mpc_n_t, umax=umax, umin=umin)
        elif control_strategy == 'mesocoup':
            avg_goal_dist, cost_val = mas.update_system_mpc_mesocoupling(Q, R, P, 
                                                                        n_t_mes=mpc_n_t, n_t_mic=mpc_n_t2, 
                                                                        rad_max=rad_max, lap_lambda=lap_lambda,
                                                                        umax=umax, umin=umin)
        else:
            raise NotImplementedError(f"Unknown control strategy '{control_strategy}'")
    solution_time = mas.control_solution_time
    #print("Total optimization time (s):", mas.control_solution_time)
    #print("Final cost:", cost_val)
    #print("Final average goal distance:", avg_goal_dist)
    return solution_time, cost_val, avg_goal_dist


if __name__ == '__main__':

    # Load config
    config_name = args.config
    with open(f'cfg/{config_name}.yaml') as f:
        experiment_parameters = yaml.load(f, Loader=yaml.FullLoader)
    exprts = list(product_dict(**experiment_parameters))

    with open(f'cfg/metaparams.yaml') as f:
        metaparams = yaml.load(f, Loader=yaml.FullLoader)
    n_exper_runs = metaparams['n_exper_runs']
    do_mp = metaparams['multiprocess']

    # Results initialization
    os.makedirs('results/', exist_ok=True)
    df_path = f'results/{config_name}.csv'
    if os.path.exists(df_path):
        df_header = False
    else:
        df_header = True

    exprt_keys = exprts[0].keys()
    df_dict = {key: [] for key in exprt_keys} | {'solution_time_MEAN': [],
                                                 'solution_time_STD': [],
                                                 'cost_val_MEAN': [],
                                                 'cost_val_STD': [],
                                                 'avg_goal_dist_MEAN': [],
                                                 'avg_goal_dist_STD': [],}

    for exprt in exprts:
        print(exprt)
        outs = []
        if do_mp:
            exprt_list = [exprt for _ in range(n_exper_runs)]
            outs = mp_kwargs_wrapper(linear_mpc, exprt_list)
        else:
            for edx in range(n_exper_runs):
                task_out = linear_mpc(**exprt)
                outs.append(task_out)
        outs = np.array(outs)
        out_means = np.mean(outs, axis=0) 
        out_stds = np.std(outs, axis=0) 
        for key in exprt_keys: 
            df_dict[key] = [exprt[key]] 
    
        df_dict['solution_time_MEAN'] = [out_means[0]]
        df_dict['solution_time_STD'] = [(out_stds[0])]
        df_dict['cost_val_MEAN'] = [(out_means[1])]
        df_dict['cost_val_STD'] = [(out_stds[1])]
        df_dict['avg_goal_dist_MEAN'] = [(out_means[2])]
        df_dict['avg_goal_dist_STD'] = [(out_stds[2])]
    
        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(df_path, mode='a', header=df_header, index=False)
        df_header = False

        