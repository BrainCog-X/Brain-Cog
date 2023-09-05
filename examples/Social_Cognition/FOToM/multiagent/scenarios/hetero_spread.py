import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_good_agents=2, num_adversaries=0):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.max_steps = 25
        num_agents = num_good_agents
        self.n_agent_a = num_agents // 2 # 2
        self.n_agent_b = num_agents // 2 # 2
        num_landmarks = num_good_agents
        world.collaborative = True
        self.agent_size = 0.10
        self.n_others = 3
        self.n_group = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.id = i
            if i < self.n_agent_a:
                agent.size = self.agent_size
                agent.accel = 3.0
                agent.max_speed = 1.0
            else:
                agent.size = self.agent_size / 2
                agent.accel = 4.0
                agent.max_speed = 1.3

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.num_steps = 0
        self.end_steps = world.max_steps
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if i < self.n_agent_a:
                agent.color = np.array([0.35, 0.35, 0.85])
            else:
                agent.color = np.array([0.35, 0.85, 0.35])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        shaped_reward = False
        if shaped_reward:  # distance-based reward
            for l in world.landmarks:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                rew -= min(dists)
            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent):
                        rew -= 1
            return rew
        else:
            win_agents = []
            for land in world.landmarks:
                for a in world.agents:
                    if self.is_collision(a, land):
                        win_agents.append(a)
                        break
            rew += 2 * len(set(win_agents))

            def bound(x):
                if x > 1.0:
                    return min(np.exp(2 * x - 2), 10)
                else:
                    return 0.0
            bound_rew = 0.0
            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                bound_rew -= bound(x)
            rew += bound_rew
            return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        other_vel = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                other_vel.append([0, 0])
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + comm +
                              other_vel + entity_pos + other_pos)
