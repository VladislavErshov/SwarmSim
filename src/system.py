import numpy as np
import hdbscan
import scipy.cluster.hierarchy as hcluster
import time
import importlib

import src.state_generator as gen
from src.clustering import epsdel_clustering
from src.optimizer import mpc_solver


# Find if performance counter module exists
PYPAPI_SPEC = importlib.util.find_spec('pypapi')
if PYPAPI_SPEC is not None:
    # pypapi = importlib.util.module_from_spec(PYPAPI_SPEC)
    from pypapi import events as papi_events, papi_high
    try:
        papi_high.start_counters([papi_events.PAPI_FP_OPS,])
    except:
        PYPAPI_SPEC = None


def _lin_sys(x, A, u, B):
    return A @ x + B @ u


def _true_cost(x0, goal, A, B, u_seq, Q, R, P, T):
    cost = 0
    x = x0
    for t in range(T):
        cost += (x - goal) @ Q @ (x - goal) + u_seq[t] @ R @ u_seq[t]
        x = _lin_sys(x, A, u_seq[t], B)
    cost += (x - goal) @ P @ (x - goal)
    return cost


class MultiAgentSystem():
    """A multi-agent system dynamics simulator."""

    def __init__(self, n_agents=1, agent_dim=1, control_dim=1, global_goal=np.array([0]),
                 state_gen=gen.random_blobs, state_gen_args=[1, 1, 1, 1],
                 clust_algo='epsdel', clust_algo_params=[1, 1], coll_d=None) -> None:
        """
        Args:
            n_agents:               Number of agents
            agent_dim:              Agent dimensionality
            control_dim:            Control input dimensionality
            global_goal:            Goal point coordinates
            state_gen:              Initial agent state generator 
            state_gen_args:         Parameters of the initial state generator
            clust_algo:             Cluster identification algorithm
            clust_algo_params:      Parameters of cluster identification (depending on the algorithm):
                ________________________________________________
                algo        | parameters
                ________________________________________________
                epsdel      | epsilon, delta
                hdbscan     | alpha, leaf_size, min_cluster_size
            coll_d:                 Agent diameter for collision avoidance 
                                    [NOTE: LEAVE IT None FOR NOW!!!]
                                    [TODO: different agent sizes]
        """
        self.n_agents = n_agents
        self.agent_dim = agent_dim
        self.control_dim = control_dim
        self.agent_states = np.zeros((n_agents, agent_dim))
        self.system_goal = global_goal
        self.agents = state_gen(LinearAgentNd, agent_dim,
                                n_agents, *state_gen_args)
        self.clust_algo = clust_algo
        self.clust_algo_params = clust_algo_params
        self.avg_goal_dist = []
        self.cvx_time = 0.
        self.cvx_time_nocoup = 0.
        self.cvx_ops = 0
        self.cvx_ops_nocoup = 0
        self.laplacian = None
        self.coll_d = coll_d
        self.do_coupling = True
        self.clusters = {}
        self._re_eval_system()

    def _re_eval_system(self, re_compute_clusters=True):
        """Re-evaluate full system state by gathering each agent states"""
        for idx, agent in self.agents.items():
            self.agent_states[idx] = agent.state
        self.avg_goal_dist.append(np.linalg.norm(
            self.agent_states - self.system_goal, axis=1).mean(axis=0))
        self._re_eval_clusters(re_compute_clusters)

    def _re_eval_clusters(self, re_compute_clusters=True):
        """Re-evaluate clusters"""
        if re_compute_clusters:
            algo = self.clust_algo
            algo_parameters = self.clust_algo_params
            if algo == 'epsdel':
                epsv, delv = algo_parameters
                self.clust_labels, _, _, lap_mat = epsdel_clustering(
                    self.agent_states, epsv, delv)
                self.laplacian = lap_mat
            elif algo == 'hdbscan':
                # TODO adjacency and Laplacian matrices
                raise NotImplementedError(
                    "Sorry! The method is not functioning at the moment. Plaese, use 'epsdel' method.")
                alpha, leaf_size, min_cluster_size = algo_parameters
                clusterer = hdbscan.HDBSCAN(alpha=alpha,
                                            leaf_size=leaf_size,
                                            min_cluster_size=min_cluster_size,
                                            min_samples=1)
                clusterer.fit(self.agent_states)
                self.clust_labels = clusterer.labels_
            else:
                raise ValueError(
                    "Cluster identification algorithm not implemented. Plaese, use 'hierarchy' method.")

            self.n_clusters = max(self.clust_labels) + 1
            self.clusters = {}
            self.cluster_n_agents = np.zeros((self.n_clusters))
            self.cluster_states = np.zeros((self.n_clusters, self.agent_dim))
            for cdx in range(self.n_clusters):
                agent_indices = np.where(self.clust_labels == cdx)[0]
                n_agents_clust = agent_indices.size
                cluster = LinearClusterNd({loc_idx: self.agents[loc_idx] for loc_idx in agent_indices},
                                          n_agents_clust,
                                          self.agent_dim)
                self.clusters[cdx] = cluster
                self.cluster_n_agents[cdx] = n_agents_clust
                self.cluster_states[cdx] = cluster.state
        else:
            self.cluster_states = np.zeros((self.n_clusters, self.agent_dim))
            for cdx, cluster in self.clusters.items():
                cluster.update_state()
                self.cluster_states[cdx] = cluster.state

    # Non-correct simplified implementation
    def update_system_descent(self, step_size=0.01):
        """
        !!! DEPRECATED
        Simple descent algorithm: cluster states are corrected 
        according to the fraction of the distance toward the goal.

        Args:
            step_size:      Gradient step size
        """
        for cluster in self.clusters:
            state = cluster.state
            meso_control = step_size * (self.system_goal - state)
            cluster.propagate_input(meso_control)
            self._re_eval_system()

    def update_system_mpc_distributed(self, Q, R, P, n_t=10, umax=None, umin=None):
        """
        !!! DEPRECATED
        'Distributed' (iterated) MPC algorithm: agent states are corrected
        according to a micro-scale controller derived by optimizing MPC cost
        for each agent iteratively and separately.

        Args:
            Q:              State-cost weight matrix
            R:              Control-cost weight matrix
            P:              Terminal-state-cost weight matrix
            n_t:            Number of time steps in MPC
            umax, umin:     Control value constraints

        Returns:
            avg_goal_dist:      Average distances toward the goal point for all agents (list of all distances along the path)
            cost_val:           Value of the cost function at the final step        
        """
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            papi_high.start_counters([papi_events.PAPI_FP_OPS,])
        time_0 = time.perf_counter()
        cost_val = 0.
        for _, agent in self.agents.items():
            A = agent.A
            B = agent.B
            x0 = agent.state
            state_dynamics, u_dynamics, cost_val_agnt = mpc_solver.conventional_solve(A, B, n_t,
                                                                                      Q, R, P, x0, self.agent_dim,
                                                                                      x_star_in=self.system_goal,
                                                                                      coll_d=self.coll_d,
                                                                                      umax=umax, umin=umin)
            cost_val += cost_val_agnt
            # for tdx in range(n_t):
            #    agent.propagate_input(u_dynamics[:, tdx])
            agent.propagate_input(u_dynamics[:, 0])
            self._re_eval_system()

        goal = np.kron(np.ones((self.n_agents)), self.system_goal)
        x0_true = self.agent_states.flatten()
        true_cost = _true_cost(x0_true, goal, A, B,
                               u_dynamics.T, Q, R, P, n_t)

        self.cvx_time += time.perf_counter() - time_0
        self.cvx_time_nocoup = self.cvx_time
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            self.cvx_ops += papi_high.stop_counters()
            self.cvx_ops_nocoup = self.cvx_ops
        cost_val /= self.n_agents
        return self.avg_goal_dist, cost_val, true_cost

    def update_system_mpc(self, Q, R, P, n_t=10, umax=None, umin=None):
        """
        Full-state MPC algorithm: agent states are corrected
        according to a micro-scale controller derived by optimizing MPC cost
        for the full system state by combining each agent state into a 
        'n_agents * agent_dim'-dimensional vector.

        Args:
            Q:              State-cost weight matrix (for a single agent)
            R:              Control-cost weight matrix (for a single agent)
            P:              Terminal-state-cost weight matrix (for a single agent)
            n_t:            Number of time steps in MPC
            umax, umin:     Control value constraints

        Returns:
            avg_goal_dist:      Average distances toward the goal point for all agents (list of all distances along the path)
            cost_val:           Value of the cost function at the final step        
        """
        A = np.zeros((self.agent_dim * self.n_agents,
                     self.agent_dim * self.n_agents))
        B = np.zeros((self.agent_dim * self.n_agents,
                     self.control_dim * self.n_agents))
        for adx, agent in self.agents.items():
            A[adx * self.agent_dim: (adx + 1) * self.agent_dim,
              adx * self.agent_dim: (adx + 1) * self.agent_dim] = agent.A
            B[adx * self.agent_dim: (adx + 1) * self.agent_dim,
              adx * self.control_dim: (adx + 1) * self.control_dim] = agent.B
        x0 = self.agent_states.flatten()
        goal = np.kron(np.ones((self.n_agents)), self.system_goal)
        Q = np.kron(np.eye(self.n_agents), Q)
        R = np.kron(np.eye(self.n_agents), R)
        P = np.kron(np.eye(self.n_agents), P)
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            papi_high.start_counters([papi_events.PAPI_FP_OPS,])
        time_0 = time.perf_counter()
        state_dynamics, u_dynamics, cost_val = mpc_solver.conventional_solve(A, B, n_t,
                                                                             Q, R, P, x0, self.agent_dim,
                                                                             x_star_in=goal,
                                                                             coll_d=self.coll_d,
                                                                             umax=umax, umin=umin)

        x0_true = self.agent_states.flatten()
        true_cost = _true_cost(x0_true, goal, A, B,
                               u_dynamics.T, Q, R, P, n_t)

        self.cvx_time += time.perf_counter() - time_0
        self.cvx_time_nocoup = self.cvx_time
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            self.cvx_ops += papi_high.stop_counters()
        for adx, agent in self.agents.items():
            # for tdx in range(n_t):
            #    agent.propagate_input(u_dynamics[adx * self.agent_dim : (adx + 1) * self.agent_dim, tdx])
            agent.propagate_input(
                u_dynamics[adx * self.agent_dim: (adx + 1) * self.agent_dim, 0])
        self._re_eval_system()
        return self.avg_goal_dist, cost_val, true_cost

    def update_system_mpc_mesoonly(self, Q, R, P, n_t=10, umax=None, umin=None):
        """
        Cluster control MPC algorithm: agent states are corrected
        according to a meso-scale controller derived by optimizing MPC cost
        for the cluster states by combining full system states state into a 
        'n_clusters * agent_dim'-dimensional vector.

        Args:
            Q:              State-cost weight matrix (for a single cluster)
            R:              Control-cost weight matrix (for a single cluster)
            P:              Terminal-state-cost weight matrix (for a single cluster)
            n_t:            Number of time steps in MPC
            umax, umin:     Control value constraints

        Returns:
            avg_goal_dist:      Average distances toward the goal point for all agents (list of all distances along the path)
            cost_val:           Value of the cost function at the final step        
        """
        A = np.zeros((self.agent_dim * self.n_clusters,
                     self.agent_dim * self.n_clusters))
        B = np.zeros((self.agent_dim * self.n_clusters,
                     self.control_dim * self.n_clusters))
        for cdx, cluster in self.clusters.items():
            A[cdx * self.agent_dim: (cdx + 1) * self.agent_dim,
              cdx * self.agent_dim: (cdx + 1) * self.agent_dim] = cluster.A
            B[cdx * self.agent_dim: (cdx + 1) * self.agent_dim,
              cdx * self.control_dim: (cdx + 1) * self.control_dim] = cluster.B
        x0 = self.cluster_states.flatten()
        goal = np.kron(np.ones((self.n_clusters)), self.system_goal)
        calpha_diag = np.diag(self.cluster_n_agents)
        Q = np.kron(calpha_diag, Q)
        R = np.kron(calpha_diag, R)
        P = np.kron(calpha_diag, P)
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            papi_high.start_counters([papi_events.PAPI_FP_OPS,])
        time_0 = time.perf_counter()
        state_dynamics, u_dynamics, cost_val = mpc_solver.conventional_solve(A, B, n_t,
                                                                             Q, R, P, x0, self.agent_dim,
                                                                             x_star_in=goal,
                                                                             coll_d=self.coll_d,
                                                                             umax=umax, umin=umin)

        x0_true = self.cluster_states.flatten()
        true_cost = _true_cost(x0_true, goal, A, B,
                               u_dynamics.T, Q, R, P, n_t)

        self.cvx_time += time.perf_counter() - time_0
        self.cvx_time_nocoup = self.cvx_time
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            self.cvx_ops += papi_high.stop_counters()
        for cdx, cluster in self.clusters.items():
            # for tdx in range(n_t):
            #    cluster.propagate_input(u_dynamics[cdx * self.agent_dim : (cdx + 1) * self.agent_dim, tdx])
            cluster.propagate_input(
                u_dynamics[cdx * self.agent_dim: (cdx + 1) * self.agent_dim, 0])
        self._re_eval_system(self.do_coupling)
        return self.avg_goal_dist, cost_val, true_cost

    def update_system_mpc_microcoupling(self, Q, R, P,
                                        n_t_mic=8, n_t_cpl=None,
                                        rad_max=10., lap_lambda=1.,
                                        umax=None, umin=None,
                                        turn_cpl_off=True):
        """
        Micro-scale control with coupling MPC algorithm: agent states are corrected
        according to a micro-scale controller derived by optimizing 
        MPC cost for the individual agent states and coupling terms. Agent states are 
        packed into a 'n_agents * agent_dim'-dimensional vector. Coupling terms
        are also packed into a 'n_clusters * agent_dim'-dimensional
        vector, introduced into a quadratic form with the system Laplacian matrix.
        The resulting functional is as follows:

            J = Sum^N_mes (x_mic^T Q x_mic + u_mic^T R_mes u_mic) + terminal_quad_form +
            +   Sum^N_cpl (x_mic^T L x_mic)

        Args:
            Q:              State-cost weight matrix (for a single cluster)
            R:              Control-cost weight matrix (for a single agent and cluster, prefer R = Identity)
            P:              Terminal-state-cost weight matrix (for a single agent)
            n_t_mic:        Number of time steps in the micro-scale part of MPC
            n_t_cpl:        Number of time steps in the coupling part of MPC
            rad_max:        Target maximum cluster radius, used to activate coupling
            lap_lambda:     Coupling weight in the cost functional
            umax, umin:     Control value constraints
            turn_cpl_off:   Turn coupling off forever when clusters with a desired rad_max achieved

        Returns:
            avg_goal_dist:      Average distances toward the goal point for all agents (list of all distances along the path)
            cost_val:           Value of the cost function at the final step        
        """
        A = np.zeros((self.agent_dim * self.n_agents,
                     self.agent_dim * self.n_agents))
        B = np.zeros((self.agent_dim * self.n_agents,
                     self.control_dim * self.n_agents))
        clust_rads = []
        if n_t_cpl is None:
            n_t_cpl = n_t_mic
        for adx, agent in self.agents.items():
            A[adx * self.agent_dim: (adx + 1) * self.agent_dim,
              adx * self.agent_dim: (adx + 1) * self.agent_dim] = agent.A
            B[adx * self.agent_dim: (adx + 1) * self.agent_dim,
              adx * self.control_dim: (adx + 1) * self.control_dim] = agent.B
        for cdx, cluster in self.clusters.items():
            clust_rads.append(cluster.rad)
        if (np.max(clust_rads) > rad_max) and (self.do_coupling):
            lap_mat_aug = np.kron(self.laplacian, np.eye(
                self.agent_dim)) / self.n_agents
        else:
            lap_mat_aug = None
            if turn_cpl_off:
                self.do_coupling = False
        x0 = self.agent_states.flatten()
        goal = np.kron(np.ones((self.n_agents)), self.system_goal)
        Q = np.kron(np.eye(self.n_agents), Q)
        R = np.kron(np.eye(self.n_agents), R)
        P = np.kron(np.eye(self.n_agents), P)
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            papi_high.start_counters([papi_events.PAPI_FP_OPS,])
        time_0 = time.perf_counter()
        state_dynamics, u_dynamics, cost_val = mpc_solver.microcoupling_solve(A, B, n_t_mic, Q, R, P, x0, self.agent_dim, n_t_cpl,
                                                                              lap_mat_aug, lap_lambda,
                                                                              x_star_in=goal, coll_d=self.coll_d,
                                                                              umax=umax, umin=umin,)
        time_1 = time.perf_counter()
        self.cvx_time += time_1 - time_0

        x0_true = self.agent_states.flatten()
        true_cost = _true_cost(x0_true, goal, A, B,
                               u_dynamics.T, Q, R, P, n_t_mic)

        if lap_mat_aug is None:
            self.cvx_time_nocoup += time_1 - time_0
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            ops = papi_high.stop_counters()
            self.cvx_ops += ops
            if lap_mat_aug is None:
                self.cvx_ops_nocoup += ops
        for adx, agent in self.agents.items():
            # for tdx in range(n_t):
            #    agent.propagate_input(u_dynamics[adx * self.agent_dim : (adx + 1) * self.agent_dim, tdx])
            agent.propagate_input(
                u_dynamics[adx * self.agent_dim: (adx + 1) * self.agent_dim, 0])
        self._re_eval_system(self.do_coupling)
        return self.avg_goal_dist, cost_val, true_cost

    def update_system_mpc_mesocoupling(self, Q, R, P,
                                       n_t_mes=8, n_t_cpl=2,
                                       rad_max=10., lap_lambda=1.,
                                       umax_mes=None, umin_mes=None,
                                       umax_cpl=None, umin_cpl=None,
                                       turn_cpl_off=True):
        """
        Cluster control with coupling MPC algorithm: agent states are corrected
        according to a meso- and micro- scale controllers derived by optimizing 
        MPC cost for the cluster states and coupling terms. Cluster states are 
        packed into a 'n_clusters * agent_dim'-dimensional vector. Coupling terms
        are regular agent states packed into a 'n_clusters * agent_dim'-dimensional
        vector, introduced into a quadratic form with the system Laplacian matrix.
        The resulting functional is as follows:

            J = Sum^N_mes (x_mes^T Q x_mes + u_mes^T R_mes u_mes) + terminal_quad_form +
            +   Sum^N_cpl (x_cpl^T L x_cpl + u_cpl^T R_cpl u_cpl)

        Args:
            Q:                      State-cost weight matrix (for a single cluster)
            R:                      Control-cost weight matrix (for a single agent and cluster, prefer R = Identity)
            P:                      Terminal-state-cost weight matrix (for a single agent)
            n_t_mes:                Number of time steps in the meso-scale part of MPC
            n_t_cpl:                Number of time steps in the coupling part of MPC
            rad_max:                Target maximum cluster radius, used to activate coupling
            lap_lambda:             Coupling weight in the cost functional
            umax_mes, umin_mes:     Meso-scale control value constraints
            umax_cpl, umin_cpl:     Coupling control value constraints
            turn_cpl_off:           Turn coupling off forever when clusters with a desired rad_max achieved

        Returns:
            avg_goal_dist:      Average distances toward the goal point for all agents (list of all distances along the path)
            cost_val:           Value of the cost function at the final step        
        """
        A_mes = np.zeros((self.agent_dim * self.n_clusters,
                         self.agent_dim * self.n_clusters))
        A_cpl = np.zeros((self.agent_dim * self.n_agents,
                         self.agent_dim * self.n_agents))
        B_mes = np.zeros((self.agent_dim * self.n_clusters,
                         self.control_dim * self.n_clusters))
        B_cpl = np.zeros((self.agent_dim * self.n_agents,
                         self.control_dim * self.n_agents))
        clust_rads = []
        for cdx, cluster in self.clusters.items():
            A_mes[cdx * self.agent_dim: (cdx + 1) * self.agent_dim,
                  cdx * self.agent_dim: (cdx + 1) * self.agent_dim] = cluster.A
            B_mes[cdx * self.agent_dim: (cdx + 1) * self.agent_dim,
                  cdx * self.control_dim: (cdx + 1) * self.control_dim] = cluster.B
            clust_rads.append(cluster.rad)
        for adx, agent in self.agents.items():
            A_cpl[adx * self.agent_dim: (adx + 1) * self.agent_dim,
                  adx * self.agent_dim: (adx + 1) * self.agent_dim] = agent.A
            B_cpl[adx * self.agent_dim: (adx + 1) * self.agent_dim,
                  adx * self.control_dim: (adx + 1) * self.control_dim] = agent.B
        if (np.max(clust_rads) > rad_max) and (self.do_coupling):
            lap_mat_aug = np.kron(self.laplacian, np.eye(
                self.agent_dim)) / self.n_agents
            # lap_mat_aug = None
        else:
            lap_mat_aug = None
            if turn_cpl_off:
                self.do_coupling = False
        x0_mes = self.cluster_states.flatten()
        x0_cpl = self.agent_states.flatten()
        goal = np.kron(np.ones((self.n_clusters)), self.system_goal)
        calpha_diag = np.diag(self.cluster_n_agents)
        Q_mes = np.kron(calpha_diag, Q)
        R_mes = np.kron(calpha_diag, R)
        P_mes = np.kron(calpha_diag, P)
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            papi_high.start_counters([papi_events.PAPI_FP_OPS,])
        time_0 = time.perf_counter()
        cl_dyn, ag_dyn, u_mes, u_cpl, cost_val = mpc_solver.mesocoupling_solve(A_mes, B_mes, n_t_mes, Q_mes, R_mes, P_mes, x0_mes, self.agent_dim,
                                                                               A_cpl, B_cpl, n_t_cpl, x0_cpl, lap_mat_aug, lap_lambda,
                                                                               x_star_in=goal, coll_d=self.coll_d,
                                                                               umax_mes=umax_mes, umin_mes=umin_mes,
                                                                               umax_cpl=umax_cpl, umin_cpl=umin_cpl,)
        time_1 = time.perf_counter()
        self.cvx_time += time_1 - time_0

        x0_true = self.agent_states.flatten()
        goal_true = np.kron(np.ones((self.n_agents)), self.system_goal)
        m = np.zeros((self.n_agents, self.n_clusters))
        for i in range(self.n_agents):
            alpha = self.clust_labels[i]
            m[i, alpha] = 1
        b_true = B_cpl @ np.kron(m, np.eye(self.agent_dim))
        a_true = A_cpl
        q_true = np.kron(np.eye(self.n_agents), Q)
        r_true = R_mes
        p_true = np.kron(np.eye(self.n_agents), P)
        true_cost = _true_cost(x0_true, goal_true, a_true, b_true,
                               u_mes.T, q_true, r_true, p_true, n_t_mes)

        if lap_mat_aug is None:
            self.cvx_time_nocoup += time_1 - time_0
        if PYPAPI_SPEC and papi_events.PAPI_FP_OPS:
            ops = papi_high.stop_counters()
            self.cvx_ops += ops
            if lap_mat_aug is None:
                self.cvx_ops_nocoup += ops
        for cdx, cluster in self.clusters.items():
            # for tdx in range(n_t):
            #    cluster.propagate_input(u_dynacpls[cdx * self.agent_dim : (cdx + 1) * self.agent_dim, tdx])
            cluster.propagate_input(
                u_mes[cdx * self.agent_dim: (cdx + 1) * self.agent_dim, 0])
        if lap_mat_aug is not None:
            for adx, agent in self.agents.items():
                agent.propagate_input(
                    u_cpl[adx * self.agent_dim: (adx + 1) * self.agent_dim, 0])
        self._re_eval_system(self.do_coupling)
        return self.avg_goal_dist, cost_val, true_cost


class LinearAgentNd:
    """Linear agent with x[t+1] = Ax[t] + Bu[t] dynamics."""

    def __init__(self,
                 A, B,
                 agent_dim=1,
                 init_state=np.zeros((1))) -> None:
        """
        Args:
            A, B:           State and control transition matrices
            agent_dim:      Agent dimensionality
            init_state:     Initial agent state   
        """
        assert init_state.size == agent_dim
        self.A = A
        self.B = B
        assert A.shape == (agent_dim, agent_dim)
        assert B.shape[1] == agent_dim
        self.state = init_state
        self.agent_dim = agent_dim

    def propagate_input(self, control_val, w_mean=0, w_std=1):
        """Receive a control action and change agent state correspondingly."""
        control = self.B @ control_val
        normal = np.random.normal(w_mean, w_std, size=self.state.shape)
        w = [control_i * w_i / 100.0 for control_i, w_i in zip(control, normal)]
        self.state = self.A @ self.state + control + w

    # !!! Prefer not to use
    def set_state(self, input_state):
        """Manually set specific agent state; avoid using it and prefer propagate_input()."""
        assert self.state.shape == input_state.shape
        self.state = input_state


class LinearClusterNd():
    """Linear cluster of agents with A = avg(A_i) and B = avg(B_i) for all agents i in the cluster."""

    def __init__(self,
                 agents,
                 n_agents,
                 agent_dim) -> None:
        """
        Args:
            agents:         Dictionary of agents combined into the cluster
            n_agents:       Number of agents in the cluster
            agent_dim:      Agent dimensionality
        """
        self.agents = agents
        assert n_agents == len(agents)
        self.n_agents = n_agents
        self.agent_states = np.zeros((n_agents, agent_dim))
        self.update_state()
        self.A = 0.
        self.B = 0.
        for agent in self.agents.values():
            self.A += agent.A
            self.B += agent.B
        self.A /= self.n_agents
        self.B /= self.n_agents

    def propagate_input(self, control_val):
        """Propagate a meso-scale control action for all agents within the cluster."""
        for agent in self.agents.values():
            agent.propagate_input(control_val)
        self.update_state()

    def update_state(self):
        """Update the centroid value according to the aggregated agent state."""
        for idx in range(self.n_agents):
            self.agent_states[idx] = list(self.agents.values())[idx].state
        self.state = np.mean(self.agent_states, axis=0)
        self.rad = np.linalg.norm(
            self.agent_states - self.state, axis=1).max(axis=0)
        return self.state, self.rad
