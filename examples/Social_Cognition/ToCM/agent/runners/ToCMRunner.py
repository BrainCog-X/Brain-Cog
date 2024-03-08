import ray
import wandb
import os
import torch
import numpy as np
from agent.workers.ToCMWorker import ToCMWorker
from environments import Env

class ToCMServer:
    def __init__(self, n_workers, env_config, controller_config, model):
        # ray.init(local_mode=True) #ray.init()
        ray.init(dashboard_port=8625, object_store_memory=8*1024*1024*1024, _memory=8*1024*1024*1024, _temp_dir='~/temp/')
        # ray.init()
        self.workers = [ToCMWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model) for worker in self.workers]

    def append(self, idx, update):
        self.tasks.append(self.workers[idx].run.remote(update))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks[:] = tasks
        del tasks
        recvs = ray.get(done_id)[0]
        return recvs


class ToCMRunner:

    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.env_config = env_config
        self.env_type = env_config.ENV_TYPE
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        self.server = ToCMServer(n_workers, env_config, controller_config, self.learner.params())  # share weight
        self.save_dir = '~/ToCM/weights/seed'\
                        + str(controller_config.random_seed) + 'num_agent_2' + '/'+ learner_config.env_name + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.pretrain = True

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10):  # 10**10 50000
        print("Start ToCM Runner!")
        cur_steps, cur_episode = 0, 0

        wandb.define_metric("steps")
        wandb.define_metric("reward", step_metric="steps")
        episode_rewards = []

        while True:

            rollout, info = self.server.run()  # control_config -> worker
            episode_rewards.append(info["reward"])
            self.learner.step(rollout)
            cur_steps += info["steps_done"]
            cur_episode += 1

            if self.env_type == Env.MPE:
                if cur_steps % 1000 == 0:
                    episode_average_rewards = np.mean(episode_rewards)
                    episode_rewards = []
                    wandb.log({'reward': episode_average_rewards, 'steps': cur_steps})
                    print('cur_steps:', cur_steps, 'total_samples:',
                          self.learner.total_samples, 'reward', episode_average_rewards)
            else:
                wandb.log({'reward': info[""
                                          "reward"], 'steps': cur_steps})

                print('cur_episode:', cur_episode, 'total_samples:',
                      self.learner.total_samples, 'reward', info["reward"])
            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break

            if cur_episode % 100 == 1 and self.pretrain:
                path = self.save_dir + str( cur_episode // 100)
                torch.save(self.learner.params()['model'], path +  '_model.pth')
                torch.save(self.learner.params()['actor'], path +  '_actor.pth')
                torch.save(self.learner.params()['critic'], path +  '_critic.pth')
                print("Save model_" + str( cur_episode // 100))
            self.server.append(info['idx'], self.learner.params())

