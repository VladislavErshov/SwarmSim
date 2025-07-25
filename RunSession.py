import numpy as np
import warnings
import yaml
# import argparse
import sys
import os
import pandas as pd

from src.system import MultiAgentSystem, PYPAPI_SPEC
import src.state_generator as gen
import src.plot.plotter as pltr
from src.multiprocessing.mp_wrapper import mp_kwargs_wrapper
from src.utils import product_dict, translation_table


# with open('cfg/seed.yaml') as f:
#    seed_data = yaml.load(f, Loader=yaml.FullLoader)
#    rnd_seed = seed_data['RND_SEED']

warnings.filterwarnings('ignore')

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config', help="Config name", required=False, default='exprt_1')
# args = parser.parse_args()
# configs = args.config.split('&')
# print(configs)

if len(sys.argv) == 1:
    configs = ['exprt_1']
else:
    configs = sys.argv[1:]


# Initialize obstacles [TODO]

def linear_mpc(
    n_agents=1000,  # number of agents
    cluster_means=[(0, 5), (0, -5)], # coordinates of initial cluster centroids for each cluster
    cluster_std=0.8,  # standard deviation for gaussian blob cluster initialization
    cluster_rad=1,  # radius for uniform blob cluster initialization
    clust_eps=1.5,  # epsilon-delta clustering parameter epsilon
    agent_dim=2,  # dimensionality of each agent
    control_dim=2,  # diemnsionality of control
    goal_state=np.array([10, 0]),  # goal point coordinates
    A=np.eye(2),  # initial matrix A (state transition) for a linear agent
    B=np.eye(2),  # initial matrix B (control transition) for a linear agent
    u_bound=None,  # control constraint absolute value
    n_steps=None,  # number of MPC iterations
    mpc_n_t=16,  # MPC horizon
    mpc_n_t2=2,  # MPC horizon for the coupling term
    rad_max=2.,  # target maximum cluster radius
    lap_lambda=1.,  # coupling weight
    coll_d=None,
    control_strategy='mesocoup',  # control strategy
    dynamics_pic_dir=None, # None if prefer not to save dynamics plots, path to the save directory otherwise
    shrink_horizon=False, # 'True' if shrink MPC horizon to the number of remaining MPC iterations
    do_coupling=True,  # 'True' if compute coupling at the start of optimization
):
    if mpc_n_t2 is None:
        mpc_n_t2 = mpc_n_t
    if n_steps is None:
        n_steps = mpc_n_t
    Q = np.eye(agent_dim) / n_agents
    R = np.eye(control_dim) / n_agents
    P = np.eye(agent_dim) / n_agents  # np.zeros((agent_dim, agent_dim))
    if u_bound is None:
        u_bound = 2 * np.linalg.norm(goal_state, 2) / mpc_n_t
    umax = u_bound
    umin = -u_bound
    umax_cpl = 1
    umin_cpl = -1
    mas = MultiAgentSystem(
        n_agents,
        agent_dim,
        control_dim,
        goal_state,
        state_gen=gen.random_blobs,
        state_gen_args=[[A], [B], cluster_means, cluster_std],
        clust_algo_params=[clust_eps, clust_eps],
        coll_d=coll_d
    )
    mas.do_coupling = do_coupling
    avg_goal_dist = mas.avg_goal_dist
    cost_vals = [np.inf]
    j0_costs = [np.inf]
    for sdx in range(n_steps):
        if shrink_horizon:
            mpc_n_t_s = min(mpc_n_t, n_steps - sdx)
            mpc_n_t2_s = min(mpc_n_t2, n_steps - sdx)
        else:
            mpc_n_t_s = mpc_n_t
            mpc_n_t2_s = mpc_n_t2
        if dynamics_pic_dir is not None:
            pltr.system_state(
                mas, goal_state, avg_goal_dist,
                cost_vals[sdx], save_path=dynamics_pic_dir + control_strategy + f'_{sdx}.png'
            )
        if control_strategy == 'micro':
            avg_goal_dist, cost_val, j0_cost = mas.update_system_mpc(
                Q, R, P,
                n_t=mpc_n_t_s, umax=umax, umin=umin
            )
        elif control_strategy == 'microdist':
            avg_goal_dist, cost_val, j0_cost = mas.update_system_mpc_distributed(
                Q, R, P,
                n_t=mpc_n_t_s, umax=umax, umin=umin
            )
        elif control_strategy == 'meso':
            avg_goal_dist, cost_val, j0_cost = mas.update_system_mpc_mesoonly(
                Q, R, P,
                n_t=mpc_n_t_s, umax=umax, umin=umin
            )
        elif control_strategy == 'microcoup':
            avg_goal_dist, cost_val, j0_cost = mas.update_system_mpc_microcoupling(
                Q, R, P,
                n_t_mic=mpc_n_t_s, n_t_cpl=mpc_n_t2_s,
                rad_max=rad_max, lap_lambda=lap_lambda,
                umax=umax, umin=umin
            )
        elif control_strategy == 'mesocoup':
            avg_goal_dist, cost_val, j0_cost = mas.update_system_mpc_mesocoupling(
                Q, R, P,
                n_t_mes=mpc_n_t_s, n_t_cpl=mpc_n_t2_s,
                rad_max=rad_max, lap_lambda=lap_lambda,
                umax_mes=umax, umin_mes=umin,
                umax_cpl=umax_cpl, umin_cpl=umin_cpl
            )
        else:
            raise NotImplementedError(
                f"Unknown control strategy '{control_strategy}'")
        cost_vals.append(cost_val)
        j0_costs.append(j0_cost)
    cvx_time = mas.cvx_time
    cvx_time_nocoup = mas.cvx_time_nocoup
    cvx_gops = mas.cvx_ops / 10e9
    cvx_gops_nocoup = mas.cvx_ops_nocoup / 10e9
    # print("Total optimization time (s):", cvx_time)
    # print("Total optimization time w/o coupling (s):", cvx_time_nocoup)
    # print("Total optimization operations (GFLOPs):", cvx_gops)
    # print("Total optimization operations w/o coupling (GFLOPs):", cvx_gops_nocoup)
    # print("Final cost:", cost_val)
    # print("Final average goal distance:", avg_goal_dist[-1])
    return cvx_time, cvx_time_nocoup, cvx_gops, cvx_gops_nocoup, cost_vals, avg_goal_dist, j0_costs


if __name__ == '__main__':
    with open(f'cfg/metaparams.yaml') as f:
        metaparams = yaml.load(f, Loader=yaml.FullLoader)
    n_exper_runs = metaparams['n_exper_runs']
    do_mp = metaparams['multiprocess']
    do_dynamics = metaparams['do_dynamics']
    do_statistics = metaparams['do_statistics']
    rnd_seed = metaparams['rnd_seed']
    if rnd_seed is not False:
        np.random.seed(rnd_seed)

    # Load config
    for config_name in configs:
        with open(f'cfg/{config_name}.yaml') as f:
            experiment_parameters = yaml.load(f, Loader=yaml.FullLoader)
        experiments = list(product_dict(**experiment_parameters))

        if PYPAPI_SPEC is not None:
            print("OP count enabled")
        else:
            print("! OP count not available")

        # Results initialization
        res_dir = f'results/{config_name}/'
        os.makedirs(res_dir, exist_ok=True)
        df_res_path = res_dir + 'statistics.csv'

        if os.path.exists(df_res_path):
            df_res_header = False
        else:
            df_res_header = True

        experiment_keys = experiments[0].keys()
        df_res_dict = {key: [] for key in experiment_keys} | {
            'cvx_time_MEAN': [],
            'cvx_time_STD': [],
            'cvx_time_nocoup_MEAN': [],
            'cvx_time_nocoup_STD': [],
            'cvx_ops_MEAN': [],
            'cvx_ops_STD': [],
            'cvx_ops_nocoup_MEAN': [],
            'cvx_ops_nocoup_STD': [],
            'cost_val_MEAN': [],
            'cost_val_STD': [],
            'avg_goal_dist_MEAN': [],
            'avg_goal_dist_STD': [],
        }

        # if do_dynamics:
        #    print("DYNAMICS RUN")
        #    dyn_exprt_microcoup = {key: val[-1] for key, val in experiment_parameters.items() if key != 'control_strategy'} | {'control_strategy': 'microcoup'}
        #    dyn_exprt_mesocoup = {key: val[-1] for key, val in experiment_parameters.items() if key != 'control_strategy'} | {'control_strategy': 'mesocoup'}
        #    print(dyn_exprt_microcoup)
        #    print(dyn_exprt_mesocoup)
        #    np.random.seed(rnd_seed)
        #    _, _, _, _, cost_microcoup, dyn_microcoup = linear_mpc(**dyn_exprt_microcoup, dynamics_pic_dir=res_dir)
        #    np.random.seed(rnd_seed)
        #    _, _, _, _, cost_mesocoup, dyn_mesocoup = linear_mpc(**dyn_exprt_mesocoup, dynamics_pic_dir=res_dir)
        #    df_dyn = pd.DataFrame.from_dict({'microcoup': dyn_microcoup, 'mesocoup': dyn_mesocoup})
        #    df_dyn.to_csv(df_dyn_path, mode='w', header=True, index=False)
        #    df_cost = pd.DataFrame.from_dict({'microcoup': cost_microcoup, 'mesocoup': cost_mesocoup})
        #    df_cost.to_csv(df_cost_path, mode='w', header=True, index=False)

        for experiment in experiments:
            print(experiment)
            e_str = str(experiment)
            e_str = e_str.translate(translation_table).replace(' ', '_')
            outs = []
            np.random.seed(rnd_seed)

            if do_dynamics:
                df_dyn_dir = res_dir + f'{e_str}/'
                os.makedirs(df_dyn_dir, exist_ok=True)
            else:
                df_dyn_dir = None
            if do_mp:
                experiment_list = [experiment for _ in range(n_exper_runs - 1)] + [experiment | {'dynamics_pic_dir': df_dyn_dir}]
                outs = mp_kwargs_wrapper(linear_mpc, experiment_list)
            else:
                for edx in range(n_exper_runs):
                    task_out = linear_mpc(**experiment, dynamics_pic_dir=df_dyn_dir)
                    outs.append(task_out)

            if do_dynamics:
                cost_vals = outs[0][-3]
                avg_goal_dist = outs[0][-2]
                j0_vals = outs[0][-1]
                df_dyn = pd.DataFrame.from_dict({'cost': cost_vals[1:], 'distance': avg_goal_dist[1:], 'j0': j0_vals[1:]})
                df_dyn.to_csv(df_dyn_dir + 'dynamics.csv', mode='w', header=True, index=False)

            if do_statistics:
                outs = np.array(
                    [(*out[:-3], out[-3][-1], out[-2][-1], out[-1][-1]) for out in outs]
                )
                out_means = np.mean(outs, axis=0)
                out_stds = np.std(outs, axis=0)
                for key in experiment_keys:
                    df_res_dict[key] = [experiment[key]]

                df_res_dict['cvx_time_MEAN'] = [out_means[0]]
                df_res_dict['cvx_time_STD'] = [(out_stds[0])]
                df_res_dict['cvx_time_nocoup_MEAN'] = [out_means[1]]
                df_res_dict['cvx_time_nocoup_STD'] = [(out_stds[1])]
                df_res_dict['cvx_ops_MEAN'] = [out_means[2]]
                df_res_dict['cvx_ops_STD'] = [(out_stds[2])]
                df_res_dict['cvx_ops_nocoup_MEAN'] = [out_means[3]]
                df_res_dict['cvx_ops_nocoup_STD'] = [(out_stds[3])]
                df_res_dict['cost_val_MEAN'] = [(out_means[4])]
                df_res_dict['cost_val_STD'] = [(out_stds[4])]
                df_res_dict['avg_goal_dist_MEAN'] = [(out_means[5])]
                df_res_dict['avg_goal_dist_STD'] = [(out_stds[5])]
                df_res_dict['j0_val_MEAN'] = [(out_means[6])]
                df_res_dict['j0_val_STD'] = [(out_stds[6])]

                df_res = pd.DataFrame.from_dict(df_res_dict)
                df_res.to_csv(df_res_path, mode='a', header=df_res_header, index=False)
                df_res_header = False
