import numpy as np
import src.engine.state_generator as gen



class AgentNd():

    def __init__(self,
                 agent_dim=1,
                 state=np.zeros((1))) -> None:
        assert state.size == agent_dim
        self.state = state
        self.agent_dim = agent_dim
        
    def micro_input(self, control_val):
        self.state += control_val

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
        self.full_cluster_state = np.zeros((n_agents, agent_dim))
        for idx in range(n_agents):
            self.full_cluster_state[idx] = agents[idx].get_state()
        self.centroid = np.mean(self.full_cluster_state, axis=0)

    def meso_input(self, control_val):
        self.full_cluster_state += control_val
        self.centroid += control_val
        for agent in self.agents:
            agent.micro_input(control_val)

    def get_centroid(self):
        return self.centroid


class MultiAgentSystem():

    def __init__(self, 
                 n_agents=1,
                 agent_dim=1,
                 global_goal=np.array([0]),
                 state_gen=gen.default_uniform_cube) -> None:
        self.agents = []
        self.n_agents = n_agents
        self.agent_dim = agent_dim
        self.full_system_state = np.zeros((n_agents, agent_dim))
        self.system_goal = global_goal
        for idx in range(n_agents):
            agent = state_gen(AgentNd, agent_dim)
            self.agents.append(agent)
            self.full_system_state[idx] = agent.get_state()
        self.clusters = [ClusterNd(self.agents, n_agents, agent_dim)]
        self.cluster_centroids = [cluster.get_centroid() for cluster in self.clusters]

    def _re_eval_full_state(self):
        for idx, agent in enumerate(self.agents):
            self.full_system_state[idx] = agent.get_state()
        self.cluster_centroids = [cluster.get_centroid() for cluster in self.clusters]

    def get_n_agents(self):
        return self.n_agents        

    def get_agents(self):
        return self.agents

    def get_full_state(self):
        return self.full_system_state

    def get_cluster_centroids(self):
        return self.cluster_centroids
    
    def update_system(self, step_size=0.01):
        # TODO: add micro- and macro-scale control
        for cluster in self.clusters:
            centroid = cluster.get_centroid()
            meso_control = step_size * (self.system_goal - centroid)
            cluster.meso_input(meso_control)
            self._re_eval_full_state()
