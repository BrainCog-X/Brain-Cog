from mpe.MPE_Env import MPEEnv


class MPE:

    def __init__(self, args):
        self.env = MPEEnv(args)  # TODO args name and random seed
        # scenario_name=args.scenario_name, benchmark=args.benchmark, num_agents=args.num_agents,
        # num_adversaries, num_landmarks, episode_length
        self.env.seed(args.seed)

        self.n_agents = self.env.num_agents
        self.agents = self.env.agents

    def to_dict(self, l):
        return {i: e for i, e in enumerate(l)}

    def step(self, action_dict):  # action dict for each agent
        # print("action_dist", action_dict)
        obs, reward, done, info = self.env.step(action_dict)  # TODO return four list
        return {i: obs[i] for i in range(self.n_agents)}, {i: reward[i] for i in range(self.n_agents)}, \
            {i: done[i] for i in range(self.n_agents)}, {i: info[i] for i in range(self.n_agents)}

    def reset(self):
        obs = self.env.reset()
        return self.to_dict(obs)

    def close(self):
        self.env.close()

    # no mask and no this usage
    def get_avail_agent_actions(self, handle):  # available handle is the i th agent, add mask
        return self.env._get_done(handle)
