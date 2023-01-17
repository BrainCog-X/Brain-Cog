# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env
import time

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Pipe


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            ob = env.render(mode='rgb_array')
            # print(len(ob), 'len(frames)')
            # print(len(ob[0]), 'len(frames[0])')
            # print(len(ob[0][0]), 'len(frames[0][0])')
            remote.send(ob)  # rgb_array
        elif cmd == 'observe':
            ob = env.observe(data)
            remote.send(ob)
        elif cmd == 'agents':
            remote.send(env.agents)
        elif cmd == 'spec':
            remote.send(env.spec)
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def observe(self, agent):
        pass

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = self.tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)    #
            return self.get_viewer().isopen

        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def get_viewer(self):
        if self.viewer is None:
            from common import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

    def tile_images(self, img_nhwc):
        """
        Tile N images into one big PxQ image
        (P,Q) are chosen to be as close as possible, and if N
        is square, then P=Q.
        input: img_nhwc, list or array of images, ndim=4 once turned into array
            n = batch index, h = height, w = width, c = channel
        returns:
            bigim_HWc, ndarray with ndim=3
        """
        img_nhwc = np.asarray(img_nhwc)
        N, h, w, c = img_nhwc.shape
        H = int(np.ceil(np.sqrt(N)))
        W = int(np.ceil(float(N) / H))
        img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
        img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
        img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
        img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
        return img_Hh_Ww_c

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs_sc: list of gym environments to run in subprocesses
        """
        # self.venv = venv
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):       # the input of step() : action
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]    # the output of step() : zip(*results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step_wait_2(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        reward, done, _cumulative_rewards = zip(*results)
        return reward, done, _cumulative_rewards

    def step_wait_3(self):
        results = [remote.recv() for remote in self.remotes]    # the output of step() : zip(*results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def agents(self):
        for remote in self.remotes:
            remote.send(('agents', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def spec(self):
        for remote in self.remotes:
            remote.send(('spec', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_images(self):
        # self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        # imgs = _flatten_list(imgs)
        return imgs

    def observe(self, agent):
        for remote, agent in zip(self.remotes, agent):
            remote.send(('observe', agent))
        return np.stack([remote.recv() for remote in self.remotes])

    # def render(self, mode='human'):
    #     return self.venv.render(mode=mode)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done):
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return