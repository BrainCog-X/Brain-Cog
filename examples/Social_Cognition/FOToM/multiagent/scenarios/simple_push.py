import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, num_good_agents=2, num_adversaries=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_good_agents + num_adversaries
        num_adversaries = num_adversaries
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
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
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[i + 1] += 0.8
            landmark.index = i
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                j = goal.index
                agent.color[j + 1] += 0.5
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def agent_reward(self, agent, world):
        '''
        Rewrite
        '''
        shaped_reward = False
        if shaped_reward:  # distance-based reward
        # the distance to the goal
            return -np.sqrt(np.sum(np.square(
                agent.state.p_pos - agent.goal_a.state.p_pos)))
        else:
            pos_rew, adv_rew = 0.0, 0.0
            for a in self.adversaries(world):
                if self.is_collision(a, agent):
                    adv_rew -= 5.0
            if self.is_collision(agent, agent.goal_a):
                pos_rew += 5.0
            rew = pos_rew + adv_rew
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

    def adversary_reward(self, agent, world):
        '''
        Rewrite
        '''
        shaped_reward = False
        if shaped_reward:  # distance-based reward
            # keep the nearest good agents away from the goal
            agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos -
                       a.goal_a.state.p_pos))) for a in world.agents if not a.adversary]
            pos_rew = min(agent_dist)
            #nearest_agent = world.good_agents[np.argmin(agent_dist)]
            #neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
            neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
            #neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
            return pos_rew - neg_rew
        else:
            rew = 0.0
            for a in self.good_agents(world):
                if self.is_collision(a, a.goal_a):
                    rew -= 5.0
                # if self.is_collision(a, agent):
                #     rew += 5.0
            if self.is_collision(agent, agent.goal_a):
                rew += 5.0
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
            if other is agent and not other.adversary:
                other_vel.append([0, 0])
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append([0, 0])

        if not agent.adversary:
            return np.concatenate([agent.state.p_vel] +
            [agent.goal_a.state.p_pos - agent.state.p_pos] +
          [agent.color] + entity_color +
                  other_vel + entity_pos + other_pos)
        else:
            #other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate([agent.state.p_vel] +
                  other_vel + entity_pos + other_pos)
