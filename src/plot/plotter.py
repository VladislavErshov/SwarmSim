import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np



rename = {
    'n_steps': 'number of max time steps',
    'n_agents': 'number of agents',
    'mpc_n_t': 'MPC max horizon',
    'micro': 'micro-scale strategy',
    'microcoup': 'micro-scale strategy',
    'mesocoup': 'meso-scale strategy',
    'control_strategy': 'control strategy',
    'solution_time': 'solution_time',
    'cvx_time': 'solution time',
    'cvx_time_nocoup': 'solution time',
    'cvx_ops': 'solution FLOPs',
    'cvx_ops_nocoup': 'solution FLOPs',
    'cost_val': 'cost value',
    'cost': 'cost value',
    'j0_val': r'$J_0$ cost value',
    'j0': r'$J_0$ cost value',
    'rad_max': r'cluster radius $\delta$',
    'cluster_rad': r'cluster radius $\delta$',
    'avg_goal_dist': 'average distance to the goal',
    'distance': 'average distance to the goal',
}

units = {
    'solution_time': 'time, seconds',
    'cvx_time': 'time, seconds',
    'cvx_time_nocoup': 'time, seconds',
    'cvx_ops': 'operations, GFLOPs',
    'cvx_ops_nocoup': 'operations, GFLOPs',
    'cost_val': 'cost value',
    'cost': 'cost value',
    'j0_val': 'cost value',
    'j0': 'cost value',
    'rad_max': 'cluster radius',
    'avg_goal_dist': r'$\ell_2$ distance',
    'distance': r'$\ell_2$ distance',
}

colors = {
    'microcoup': 'b',
    'mesocoup': 'r',
}

nonestring = 'None'


def system_state(mas, goal_state, avg_goal_dist, cost_val, show=False, save_path=None):
    """
    Plot system state.

    Args:
        mas:            Multi-agent system simulator
        goal_state:     System goal coordinates
        avg_goal_dist:  Distance from agents to the goal (averaged)
        cost_val:       Cost functional value
    """
    fstate = mas.agent_states
    cluster_states = mas.cluster_states
    n_clusters = mas.n_clusters
    clust_labels = mas.clust_labels
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    ax.scatter(goal_state[0], goal_state[1], s=30, c='k', marker='x')
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    for cdx in range(n_clusters):
        agent_indices = np.where(clust_labels == cdx)[0]
        ax.scatter(fstate[agent_indices, 0], fstate[agent_indices, 1], 
                    s=5, c=colors[cdx], marker='.')
        ax.scatter(cluster_states[cdx][0], cluster_states[cdx][1], 
                    s=40, facecolors='none', edgecolors='#000000', marker='o')
    ax.set_title(f"Avg goal dist: {avg_goal_dist[-1]:.2}; cost: {cost_val:.2f}")
    ax.set_xlim(-5, goal_state[0] * 1.2)
    ax.set_ylim(-10, 10)
    if show:
        plt.show() 
    if save_path is not None:
        plt.savefig(save_path)


def exprt_results(dfs_perstrat, param_col, xvalues, 
                  mean_cols, std_cols, 
                  xscale='log', yscale='log', xlogbase=10, ylogbase=10, 
                  save_dir=None):
    #colors = cm.rainbow(np.linspace(0, 1, len(list(dfs_perstrat.keys()))))
    plt.rcParams.update({'font.size': 14})
    for means, stds in zip(mean_cols, std_cols):
        data_name = means[:-5]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        for sdx, (strat, df) in enumerate(dfs_perstrat.items()):
            ax.errorbar(xvalues, df[means], df[stds], 
                        c=colors[strat], elinewidth=1, capsize=2, capthick=1,
                        label=rename[strat])
            ax.grid('major')
            if xscale == 'log':
                ax.set_xscale(xscale, base=xlogbase)
            else:
                ax.set_xscale(xscale)
            if yscale == 'log':
                ax.set_yscale(yscale, base=ylogbase)
            else:
                ax.set_yscale(yscale)
            #ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            #ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_title(f'{rename[data_name]}: variable {rename[param_col]}')
            ax.set_ylabel(units[data_name])
            ax.set_xlabel(rename[param_col])
            ax.legend()
        #plt.show()
        if save_dir is not None:
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            plt.savefig(save_dir + f'{data_name}_{param_col}.png')#, bbox_inches='tight')


def exprt_dynamics(data, nd, info_string, info_string_simple, save_dir):
    x = np.arange(nd) + 1
    plt.rcParams.update({'font.size': 14})
    for ftr, subdat in data.items():
        #colors = cm.rainbow(np.linspace(0, 1, len(list(subdat.keys()))))
        fig, ax = plt.subplots(figsize=(6, 2), dpi=140)
        for sdx, (strat, ssdat) in enumerate(subdat.items()):
            ax.plot(x[3:], ssdat[3:], 
                    c=colors[strat], linewidth=1,
                    label=rename[strat])
        ax.grid('major')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(rename[ftr] + ', ' + info_string)
        ax.set_ylabel(units[ftr])
        ax.set_xlabel(r'$\tau$')
        ax.legend()
        #plt.show()
        if save_dir is not None:
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            plt.savefig(save_dir + f'{ftr}_{info_string_simple}.png')#, bbox_inches='tight')