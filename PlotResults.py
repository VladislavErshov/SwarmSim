import pandas as pd
import src.plot.plotter as pltr



if __name__ == '__main__':
    exper_name = 'exprt_6'
    xscale = 'log'
    xbase = 2
    results_df = pd.read_csv(f'results/{exper_name}/statistics.csv')
    control_strats = tuple(set(results_df['control_strategy']))
    dfs_perstrat = {}
    for strat in control_strats:
        dfs_perstrat[strat] = results_df[results_df['control_strategy'] == strat].drop(['control_strategy'], axis=1)
    
    all_cols = list(list(dfs_perstrat.values())[0].columns)
    mean_cols = [col for col in all_cols if 'MEAN' in col]
    std_cols = [col for col in all_cols if 'STD' in col]
    data_cols = mean_cols + std_cols
    param_cols = [col for col in all_cols if col not in data_cols]
    n_paramators = len(param_cols)
    unique_param_vals = {col: sorted(list(set(results_df[col]))) for col in param_cols}

    for param_col in param_cols:
        dfs_currparam = {}
        rest_param_cols = [col for col in param_cols if col != param_col]
        for ri_col in rest_param_cols:
            for strat, df in dfs_perstrat.items():
                dfs_currparam[strat] = df[df[ri_col] == unique_param_vals[ri_col][-1]][data_cols]#.drop([ri_col], axis=1)
        pltr.exprt_results(dfs_currparam, param_col, unique_param_vals[param_col], mean_cols, std_cols, xscale, xbase, save_dir=f'results/{exper_name}/')
