import pandas as pd
import src.plot.plotter as pltr
import sys

from src.utils import product_dict, translation_table



if __name__ == '__main__':
    exper_name = 'exprt_8'
    subname = 'n_agents_100_mpc_n_t_16_do_coupling_False_cluster_rad_0.001_control_strategy'
    info_string = r'$\delta = 0.001$'
    info_string_simple = 'd_0.001'
    #subname = 'n_agents_100_mpc_n_t_16_do_coupling_False_cluster_rad_1_control_strategy'
    #info_string = r'$\delta = 1$'
    #info_string_simple = 'd_1'
    strats = ['microcoup', 'mesocoup']
    dfs = {}
    for strat in strats:
        df = pd.read_csv(f'results/{exper_name}/{subname}_{strat}/dynamics.csv')
        dfs[strat] = df
        features = list(df.columns)
    data = {}
    for ftr in features:
        data[ftr] = {}
        for strat in strats:
            d = dfs[strat][ftr].tolist()
            data[ftr][strat] = d
            nd = len(d)

    pltr.exprt_dynamics(data, nd, info_string, info_string_simple, f'results/{exper_name}/')



