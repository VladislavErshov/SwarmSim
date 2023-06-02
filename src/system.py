import numpy as np
import src.state_generator as gen
from src.opt import mpc_solver
import hdbscan



class LinearAgentNd():

    def __init__(self,
                 A, B,
                 agent_dim=1,
                 state=np.zeros((1))) -> None:
        assert state.size == agent_dim
        self.A = A
        self.B = B
        assert A.shape == (agent_dim, agent_dim)
        assert B.shape[1] == agent_dim
        self.state = state
        self.agent_dim = agent_dim
        
    def propagate_input(self, control_val):
        self.state = self.A @ self.state + self.B @ control_val

    def set_state(self, new_state):
        assert self.state.shape == new_state.shape
        self.state = new_state

    def get_state(self):
        return self.state

    def get_dim(self):
        return self.agent_dim


class ClusterNd():

    def __init__(self,
                 agents,
                 n_agents,
                 agent_dim) -> None:
        self.agents = agents
        self.n_agents = n_agents
        self.full_cluster_state = np.zeros((n_agents, agent_dim))
        self.centroid = self.update_centroid()

    def propagate_input(self, control_val):
        for agent in self.agents.values():
            agent.propagate_input(control_val)
        self.update_centroid()

    def update_centroid(self):
        for idx in range(self.n_agents):
            self.full_cluster_state[idx] = list(self.agents.values())[idx].get_state()
        self.centroid = np.mean(self.full_cluster_state, axis=0)
        return self.centroid


class MultiAgentSystem():

    def __init__(self, n_agents=1, agent_dim=1, control_dim=1, global_goal=np.array([0]),
                 state_gen=gen.random_blobs, state_gen_args=[1, 1, 1, 1, (-10, 10)],
                 clust_algo='hdbscan', clust_parameters=[1., 40, 5]) -> None:
        self.n_agents = n_agents
        self.agent_dim = agent_dim
        self.control_dim = control_dim
        self.full_system_state = np.zeros((n_agents, agent_dim))
        self.system_goal = global_goal
        self.agents = state_gen(LinearAgentNd, agent_dim, n_agents, *state_gen_args)
        self._re_eval_system(clust_algo, clust_parameters)

    def _re_eval_clusters(self, algo='hdbscan', algo_parameters=[1., 40, 5]):
        if algo == 'hdbscan':
            alpha, leaf_size, min_cluster_size = algo_parameters
            clusterer = hdbscan.HDBSCAN(alpha=alpha, leaf_size=leaf_size, min_cluster_size=min_cluster_size)
            clusterer.fit(self.full_system_state)
            self.clust_labels = clusterer.labels_
            self.n_clusters = max(self.clust_labels) + 1
            self.clusters = {}
            self.cluster_centroids = {}
            for cdx in range(self.n_clusters):
                agent_indices = np.where(self.clust_labels == cdx)
                n_agents_clust = agent_indices.size
                cluster = ClusterNd((dict(loc_idx, self.agents[loc_idx]) for loc_idx in agent_indices),
                                    n_agents_clust,
                                    self.agent_dim)
                self.clusters[cdx] = cluster 
                self.cluster_centroids[cdx] = cluster.centroid
        # TODO epsdel
        #elif algo == 'epsdel':
        #    epsv, delv = clust_parameters
        #    assert delv <= epsv
        else:
            raise ValueError("Cluster identification algorithm not implemented")
        pass

    def _re_eval_system(self, clust_algo, clust_parameters):
        for idx, agent in enumerate(self.agents):
            self.full_system_state[idx] = agent.get_state()
        self._re_eval_clusters(clust_algo, clust_parameters)

    def get_n_agents(self):
        return self.n_agents        

    def get_agents(self):
        return self.agents

    def get_full_state(self):
        return self.full_system_state

    def get_cluster_centroids(self):
        return self.cluster_centroids
    
    # Non-correct simplified implementation
    def update_system_simplified(self, step_size=0.01):
        for cluster in self.clusters:
            centroid = cluster.get_centroid()
            meso_control = step_size * (self.system_goal - centroid)
            cluster.propagate_input(meso_control)
            self._re_eval_system()

    def update_system_mpc(self, Q, R, P, n_t=10):
        A = np.zeros((self.agent_dim * self.n_agents, self.agent_dim * self.n_agents))
        B = np.zeros((self.agent_dim * self.n_agents, self.control_dim * self.n_agents))
        for adx, agent in self.agents.items():
            A = agent.A
            B = agent.B
            x0 = agent.state
            new_state, u = mpc_solver.use_modeling_tool(A, B, n_t, 
                                                        Q, R, P, x0, 
                                                        x_star_in=self.system_goal)
            #print(agent.state)
            #print(new_state[:, -1])
            #print(u)
            agent.set_state(new_state[:, -1])
            self._re_eval_system()


