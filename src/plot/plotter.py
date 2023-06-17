import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



rename = {
    'n_steps': 'number of max time steps',
    'n_agents': 'number of agents',
    'mpc_n_t': 'MPC max horizon',
    'micro': 'micro-scale strategy',
    'microcoup': 'micro-scale strategy with coupling',
    'mesocoup': 'meso-scale strategy with coupling',
    'control_strategy': 'control strategy',
    'solution_time': 'solution_time',
    'cvx_time': 'solution time',
    'cvx_ops': 'solution FLOPs',
    'cost_val': 'cost value',
    'avg_goal_dist': 'average distance to the goal',
}

units = {
    'solution_time': 'seconds',
    'cvx_time': 'seconds',
    'cvx_ops': 'GFLOPs',
    'cost_val': 'cost',
    'avg_goal_dist': r'$\ell_2$ distance',
}


def system_state(mas, goal_state, avg_goal_dist, cost_val):
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
    ax.set_title(f"Avg goal dist: {avg_goal_dist}; cost: {cost_val}")
    ax.set_xlim(-5, goal_state[0] * 1.2)
    ax.set_ylim(-10, 10)
    plt.show() 


def exprt_results(dfs_perstrat, param_col, xvalues, mean_cols, std_cols):
    colors = cm.rainbow(np.linspace(0, 1, len(list(dfs_perstrat.keys()))))
    for means, stds in zip(mean_cols, std_cols):
        data_name = means[:-5]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        for sdx, (strat, df) in enumerate(dfs_perstrat.items()):
            ax.errorbar(xvalues, df[means], df[stds], 
                        c=colors[sdx], elinewidth=1, capsize=2, capthick=1,
                        label=rename[strat])
            ax.grid('major')
            ax.set_title(f'{rename[data_name]}: variable {rename[param_col]}')
            ax.set_ylabel(f'value, {units[data_name]}')
            ax.set_xlabel(rename[param_col])
            ax.legend()
        plt.show()
