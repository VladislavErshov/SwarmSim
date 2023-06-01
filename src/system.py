import numpy as np
import src.state_generator as gen
from src.opt import mpc_solver



class AgentNd():

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
        
    def micro_input(self, control_val):
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
        for idx in range(n_agents):
            self.full_cluster_state[idx] = agents[idx].get_state()
        self.centroid = np.mean(self.full_cluster_state, axis=0)

    def meso_input(self, control_val):
        self.full_cluster_state += control_val
        self.centroid += control_val
        for agent in self.agents:
            agent.micro_input(control_val)

    def update_centroid(self):
        for idx in range(self.n_agents):
            self.full_cluster_state[idx] = self.agents[idx].get_state()
        self.centroid = np.mean(self.full_cluster_state, axis=0)
        return self.centroid


class MultiAgentSystem():

    def __init__(self, 
                 n_agents=1,
                 agent_dim=1,
                 global_goal=np.array([0]),
                 state_gen=gen.uniform_cube,
                 state_gen_args=[2, 1, 1]) -> None:
        self.agents = []
        self.n_agents = n_agents
        self.agent_dim = agent_dim
        self.full_system_state = np.zeros((n_agents, agent_dim))
        self.system_goal = global_goal
        for idx in range(n_agents):
            agent = state_gen(AgentNd, agent_dim, *state_gen_args)
            self.agents.append(agent)
            self.full_system_state[idx] = agent.get_state()
        self.clusters = [ClusterNd(self.agents, n_agents, agent_dim)]
        self.cluster_centroids = [cluster.update_centroid() for cluster in self.clusters]

    def _re_eval_full_state(self):
        for idx, agent in enumerate(self.agents):
            self.full_system_state[idx] = agent.get_state()
        self.cluster_centroids = [cluster.update_centroid() for cluster in self.clusters]

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
        # TODO: add micro- and macro-scale control
        for cluster in self.clusters:
            centroid = cluster.get_centroid()
            meso_control = step_size * (self.system_goal - centroid)
            cluster.meso_input(meso_control)
            self._re_eval_full_state()


class LinearMAS(MultiAgentSystem):

    def __init__(self, 
                 Q, R, P,
                 n_agents=1, 
                 agent_dim=1, 
                 global_goal=np.array([0]), 
                 state_gen=gen.uniform_cube,
                 state_gen_args=[2, 1, 1]) -> None:
        super().__init__(n_agents, agent_dim, global_goal, state_gen, state_gen_args)
        self.Q = Q
        self.R = R
        self.P = P

    def update_system_mpc(self, n_t=10):
        for agent in self.agents:
            A = agent.A
            B = agent.B
            x0 = agent.state
            new_state, u = mpc_solver.use_modeling_tool(A, B, n_t, 
                                                        self.Q, self.R, self.P, x0, 
                                                        x_star_in=self.system_goal)
            #print(agent.state)
            #print(new_state[:, -1])
            #print(u)
            agent.set_state(new_state[:, -1])
            self._re_eval_full_state()


