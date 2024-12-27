import copy
from enum import IntEnum
import numpy as np
import gymnasium as gym
import imageio

from window import Window


class HumanVatGoalEnv(gym.Env):
    """General HumanVatGoalEnv Class"""

    class Actions(IntEnum):
        noop = 0
        left = 1
        right = 2
        up = 3
        down = 4
        smash = 5  # destroying all surrounding vat(s) at same time
        pass

    class Cells(IntEnum):
        empty = 0
        wall = 1
        goal = 2
        vat = 3
        pass

    class CellsRender(object):
        # empty = np.full(shape=(64, 64, 4), fill_value=255)
        wall = imageio.imread('./materials/wall.png')
        goal = imageio.imread('./materials/goal.png')
        vat = imageio.imread('./materials/vat.png')
        agent = imageio.imread('./materials/agent.png')
        human = imageio.imread('./materials/adult.png')

    class Params(object):
        """Params for the Environment"""

        def __init__(
                self,
                map_shape=(7, 5),
                agent_pos=(1, 2),
                human_pos=((2, 3),),
                vat_pos=((3, 2), (2, 3),),
                goal_pos=((-2, 2),),
                wall_pos=(),  # user-defined walls (default surrounding walls are NOT included here)
                max_steps=50,
                env_name=None,
                # human_policy = "noop"
        ):
            self.map_shape = map_shape
            self.agent_pos = agent_pos
            self.human_pos = human_pos
            self.vat_pos = vat_pos
            self.goal_pos = goal_pos
            self.wall_pos = wall_pos
            self.max_steps = max_steps
            self.env_name = env_name

    def __init__(self, env_params=Params()):
        super(HumanVatGoalEnv, self).__init__()
        self.p = env_params

        self.actions = HumanVatGoalEnv.Actions
        self.cells = HumanVatGoalEnv.Cells

        self.action_space = gym.spaces.Discrete(len(self.actions))
        # observation_dim: dimension of observation space
        #   With fixed human_pos, currently we only consider: each vat state (broken or not) + agent_pos (x, y)
        #   e.g. [2,2,7,5] means vat1*vat2*agent_y*agent_x
        self.observation_dim = [2] * len(self.p.vat_pos) + list(self.p.map_shape)
        self.observation_space = gym.spaces.Discrete(np.prod(self.observation_dim))

        self.window = None

        if self.p.env_name is not None:
            self.descr = self.p.env_name
        else:
            n_human = len(self.p.human_pos)
            n_vat = len(self.p.vat_pos)
            n_goal = len(self.p.goal_pos)
            self.descr = self.__class__.__name__.lower()
            self.descr = self.descr.replace("human", "human" + str(n_human) + "-")
            self.descr = self.descr.replace("vat", "vat" + str(n_vat) + "-")
            self.descr = self.descr.replace("goal", "goal" + str(n_goal) + "-")

        self.num_vats = len(self.p.vat_pos)
        self.num_humans = len(self.p.human_pos)  # for empathy qlearning

        self.reset()
        return

    def reset(self):
        # reset env state
        self._gen_map()

        # reset agent & human state
        self.agent_pos = np.array(self.p.agent_pos)
        self.human_pos = [np.array(pos) for pos in self.p.human_pos]

        # reset episode statistics
        self.step_count = 0
        self.total_reward = 0
        # self.total_hidden_reward = 0  ##TODO
        # self.total_human_rewards = [0]*len(self.p.human_pos)  ##TODO

        # generate observation from state
        obs = self._gen_obs()
        return obs

    def _gen_map(self):
        self.map = np.full(shape=self.p.map_shape, fill_value=self.cells.empty)

        # place default surrounding walls
        self.map[0] = self.cells.wall
        self.map[-1] = self.cells.wall
        self.map[:, 0] = self.cells.wall
        self.map[:, -1] = self.cells.wall

        # place user-defined walls
        for pos in self.p.wall_pos:
            self.map[pos] = self.cells.wall

        # place goals
        for pos in self.p.goal_pos:
            self.map[pos] = self.cells.goal

        # place vats
        for pos in self.p.vat_pos:
            self.map[pos] = self.cells.vat

        return

    def _gen_obs(self):
        # internal state
        s_env = [(self.map[pos] == self.cells.vat) for pos in self.p.vat_pos]
        s_agent = list(self.agent_pos)
        s_human = [list(pos) for pos in self.human_pos]

        # external observation
        obs_agent = self._encode(s_env + s_agent)
        obs_human = [self._encode(s_env + s_h) for s_h in s_human]

        return [obs_agent, *obs_human]

    def _encode(self, obs):
        i = 0
        for idx, dim in enumerate(self.observation_dim):
            i *= dim
            i += obs[idx]
        assert 0 <= i <= self.observation_space.n
        return i

    def _decode(self, i):
        out = []
        for dim in reversed(self.observation_dim):
            out.append(i % dim)
            i = i // dim
        assert i == 0
        return list(reversed(out))

    def render(self, mode="window", cell_size=64, style="realistic"):
        if mode == "rgb_array":
            return self._gen_img(cell_size, style)

        elif mode == "window":
            if not isinstance(self.window, Window):
                self.window = Window(self.descr)

            if self.window.is_open():
                img = self._gen_img(cell_size, style)
                self.window.show_img(img)
                self.window.show(block=False)
                return

    def _gen_img(self, cell_size, style):
        if style == "abstract":

            h, w = self.map.shape
            img = np.full(shape=(h * cell_size, w * cell_size, 3), fill_value=255)

            def draw_cell(cell_type, cell_pos, cell_size):
                if cell_type == self.cells.empty:
                    pass
                elif cell_type == self.cells.wall:
                    x, y = np.array(cell_pos) * cell_size
                    img[x: (x + cell_size), y: (y + cell_size), :] = np.array(
                        [128, 128, 128]
                    )
                elif cell_type == self.cells.goal:
                    x, y = np.array(cell_pos) * cell_size
                    img[x: (x + cell_size), y: (y + cell_size), :] = np.array([0, 255, 0])
                elif cell_type == self.cells.vat:
                    x, y = np.array(cell_pos) * cell_size
                    img[x: (x + cell_size), y: (y + cell_size), :] = np.array([255, 0, 0])
                else:
                    pass

            def draw_agent(pos, cell_size):
                # draw rectangle
                x, y = np.array(pos) * cell_size
                img[
                int(x + 0.2 * cell_size): int(x + 0.8 * cell_size + 1),
                int(y + 0.2 * cell_size): int(y + 0.8 * cell_size + 1),
                :,
                ] = np.array([0, 0, 0])

                # # draw cicle
                # def fill_circle(img, cx, cy, r, color):
                #     h, w = img.shape[0:2]
                #     X, Y = np.ogrid[:h, :w]
                #     mask = (X-cx)**2+(Y-cy)**2 <= r**2
                #     img[mask] = color
                #     # return img
                # x0, y0 = np.array(pos) * cell_size
                # sub_img = img[int(x0):int(x0+cell_size), int(y0):int(y0+cell_size), :]
                # cx, cy, r = np.array([0.5, 0.5, 0.3]) * cell_size
                # color = np.array([0,0,0])
                # fill_circle(sub_img, cx, cy, r, color)
                pass

            def draw_human(pos, cell_size):
                # # draw rectangle
                # x, y = np.array(pos) * cell_size
                # img[int(x+0.2*cell_size):int(x+0.8*cell_size+1),
                #     int(y+0.2*cell_size):int(y+0.8*cell_size+1),:] = np.array([255,255,0])

                # draw cicle
                def fill_circle(img, cx, cy, r, color):
                    h, w = img.shape[0:2]
                    X, Y = np.ogrid[:h, :w]
                    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
                    img[mask] = color
                    # return img

                x0, y0 = np.array(pos) * cell_size
                sub_img = img[
                          int(x0): int(x0 + cell_size), int(y0): int(y0 + cell_size), :
                          ]
                cx, cy, r = np.array([0.5, 0.5, 0.36]) * cell_size
                color = np.array([255, 255, 0])
                fill_circle(sub_img, cx, cy, r, color)
                pass

            def draw_gridline(cell_size):
                img[::cell_size, :] = np.array([255, 255, 255])
                img[-1::-cell_size, :] = np.array([255, 255, 255])
                img[:, ::cell_size] = np.array([255, 255, 255])
                img[:, -1::-cell_size] = np.array([255, 255, 255])
                pass

            for i in range(h):
                for j in range(w):
                    draw_cell(self.map[i, j], (i, j), cell_size)
            for pos in self.human_pos:
                draw_human(list(pos), cell_size)
            draw_agent(self.agent_pos, cell_size)
            draw_gridline(cell_size)

            return img.astype(np.uint8)

        elif style == "realistic":

            cell_size = 64  ##In realistic mode, we fix cell_size to 64 to avoid resize of image
            h, w = self.map.shape
            img = np.full(shape=(h * cell_size, w * cell_size, 4), fill_value=255)

            def draw_cell_realistic(cell_type, cell_pos, cell_size):
                if cell_type == self.cells.empty:
                    # img_paste(cell_pos, cell_size, HumanVatGoalEnv.CellsRender.empty)
                    pass
                elif cell_type == self.cells.wall:
                    img_paste(cell_pos, cell_size, HumanVatGoalEnv.CellsRender.wall)
                elif cell_type == self.cells.goal:
                    img_paste(cell_pos, cell_size, HumanVatGoalEnv.CellsRender.goal)
                elif cell_type == self.cells.vat:
                    img_paste(cell_pos, cell_size, HumanVatGoalEnv.CellsRender.vat)
                else:
                    pass

            def draw_agent_realistic(pos, cell_size):
                img_paste(pos, cell_size, HumanVatGoalEnv.CellsRender.agent)
                pass

            def draw_human_realistic(pos, cell_size):
                img_paste(pos, cell_size, HumanVatGoalEnv.CellsRender.human)
                pass

            ##paste a png(RGBA) image on to existing img depending on the alpha channel of img_in
            def img_paste(pos, cell_size, img_in):
                x, y = np.array(pos) * cell_size
                img[x:(x + cell_size), y:(y + cell_size), 0:3][img_in[:, :, 3] > 128] = img_in[:, :, 0:3][img_in[:, :, 3] > 128]

            def draw_gridline_realistic(cell_size):
                img[::cell_size, :] = np.array([255, 255, 255, 0])
                img[-1::-cell_size, :] = np.array([255, 255, 255, 0])
                img[:, ::cell_size] = np.array([255, 255, 255, 0])
                img[:, -1::-cell_size] = np.array([255, 255, 255, 0])
                pass

            for i in range(h):
                for j in range(w):
                    draw_cell_realistic(self.map[i, j], (i, j), cell_size)
            for pos in self.human_pos:
                draw_human_realistic(list(pos), cell_size)
            draw_agent_realistic(self.agent_pos, cell_size)
            draw_gridline_realistic(cell_size)

            return img.astype(np.uint8)

    def close(self):
        if isinstance(self.window, Window):
            self.window.close()
            self.window = None
        pass

    def step(self, action):
        def apply_env_dynamics(cur_pos, action):
            assert type(cur_pos) == np.ndarray

            reward = -0.01

            cur_cell = self.map[tuple(cur_pos)]
            if cur_cell == self.cells.vat:  # got trapped in vat
                next_pos = cur_pos
                # reward += -0.05  ##TODO: should we give explicit penalty here?
            else:
                next_pos = copy.deepcopy(cur_pos)
                if action == self.actions.left:
                    next_pos += [0, -1]
                elif action == self.actions.right:
                    next_pos += [0, +1]
                elif action == self.actions.up:
                    next_pos += [-1, 0]
                elif action == self.actions.down:
                    next_pos += [+1, 0]
                elif action == self.actions.noop:
                    next_pos = next_pos
                elif action == self.actions.smash:
                    next_pos = next_pos
                    smash_pos = next_pos + [[0, -1], [0, +1], [-1, 0], [+1, 0]]
                    for sp in smash_pos:
                        smash_cell = self.map[tuple(sp)]
                        if smash_cell == self.cells.vat:
                            # agent will smash all surrounding vat(s)
                            self.map[tuple(sp)] = self.cells.empty
                            # self.total_hidden_reward -= 0.2  ##TODO
                            pass
                        pass
                    pass
                else:
                    raise

                next_cell = self.map[tuple(next_pos)]
                if next_cell == self.cells.empty:
                    next_pos = next_pos
                elif next_cell == self.cells.wall:
                    next_pos = cur_pos
                elif next_cell == self.cells.goal:
                    next_pos = next_pos
                    reward += 1.0  ##TODO: differnet rewards for reaching different goals
                elif next_cell == self.cells.vat:
                    next_pos = next_pos
                else:
                    raise

            return next_pos, reward

        self.agent_pos, reward = apply_env_dynamics(self.agent_pos, action)

        ##TODO: add more human dynamics here
        for i, _ in enumerate(self.human_pos):
            self.human_pos[i], _ = apply_env_dynamics(
                self.human_pos[i], self.actions.noop
            )  ##TODO: add different human policy
            ##TODO: human reward may be different with that of agent

        self.step_count += 1
        done = (self.step_count >= self.p.max_steps) or (
                self.map[tuple(self.agent_pos)] == self.cells.goal
        )

        obs = self._gen_obs()

        self.total_reward += reward
        info = {"total_reward": round(self.total_reward, 2)}

        return obs, reward, done, info


class BasicGoalEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(7, 5),
                agent_pos=(1, 2),
                human_pos=(),
                vat_pos=(),
                goal_pos=((-2, 2),),
                wall_pos=(),  # user-defined walls (default surrounding walls are NOT included here)
                max_steps=50,
                env_name="basic-1goal-env",
            )
        )


class BasicVatGoalEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(7, 5),
                agent_pos=(1, 2),
                human_pos=(),
                vat_pos=((3, 2),),
                goal_pos=((-2, 2),),
                wall_pos=(),  # user-defined walls (default surrounding walls are NOT included here)
                max_steps=50,
                env_name="basic-1vat-1goal-env",
            )
        )


class BasicHumanVatGoalEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(7, 5),
                agent_pos=(1, 2),
                human_pos=((3, 2),),
                vat_pos=((3, 2),),
                goal_pos=((-2, 2),),
                wall_pos=(),  # user-defined walls (default surrounding walls are NOT included here)
                max_steps=50,
                env_name="basic-1human-1vat-1goal-env",
            )
        )


class CShapeVatGoalEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(7, 5),
                agent_pos=(1, 3),
                human_pos=(),
                vat_pos=((3, 2), (3, 3)),
                goal_pos=((-2, 3),),
                wall_pos=(),  # user-defined walls (default surrounding walls are NOT included here)
                max_steps=50,
                env_name="C-shape-2vat-1goal-env",
            )
        )


class CShapeHumanVatGoalEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(7, 5),
                agent_pos=(1, 3),
                human_pos=((3, 2),),
                vat_pos=((3, 2), (3, 3)),
                goal_pos=((-2, 3),),
                wall_pos=(),  # user-defined walls (default surrounding walls are NOT included here)
                max_steps=50,
                env_name="C-shape-1human-2vat-1goal-env",
            )
        )


class SShapeVatGoalEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(10, 7),
                agent_pos=(1, 1),
                human_pos=(),
                vat_pos=((3, 1), (3, 2), (3, 3), (6, 3), (6, 4), (6, 5)),
                goal_pos=((-2, -2),),
                wall_pos=(),
                max_steps=100,
                env_name="S-shape-6vat-1goal-env",
            )
        )


class SideHumanVatGoalEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(7, 5),
                agent_pos=(1, 1),
                human_pos=((3, 3),),
                vat_pos=((3, 3),),
                goal_pos=((5, 1),),
                wall_pos=(),  # user-defined walls (default surrounding walls are NOT included here)
                max_steps=50,
                env_name="side-1human-1vat-1goal-env",
            )
        )


class SmashAndDetourEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(7, 5),
                agent_pos=(1, 1),
                human_pos=((2, 3),),
                vat_pos=((2, 3), (3, 2), (3, 3),),
                goal_pos=((5, 3),),
                wall_pos=(),  # user-defined walls (default surrounding walls are NOT included here)
                max_steps=50,
                env_name="side-1human-1vat-1goal-env",
            )
        )


class CmpxHumanVatGoalEnv(HumanVatGoalEnv):
    def __init__(self):
        super().__init__(
            env_params=HumanVatGoalEnv.Params(
                map_shape=(10, 7),
                agent_pos=(1, 3),
                human_pos=((4, 2), (5, 5), (7, 2)),
                vat_pos=((2, 3), (3, 1), (4, 1), (4, 2), (5, 5), (6, 4), (6, 5)),
                goal_pos=((-2, -2),),
                wall_pos=((1, 1), (8, 1), (8, 2)),
                max_steps=100,
                env_name="complex-3human-7vat-1goal-env",
            )
        )


env_list = ['BasicGoalEnv',
            'BasicVatGoalEnv', 'BasicHumanVatGoalEnv', 'SideHumanVatGoalEnv',
            'CShapeVatGoalEnv', 'CShapeHumanVatGoalEnv',
            'SShapeVatGoalEnv', 'SmashAndDetourEnv',
            'CmpxHumanVatGoalEnv']

if __name__ == "__main__":

    import time

    params = HumanVatGoalEnv.Params()
    params.map_shape = (9, 7)
    params.agent_pos = (1, 4)
    params.human_pos = ((3, 2), (2, 3), (2, 1))
    params.vat_pos = ((3, 2), (2, 3), (5, 2))
    params.goal_pos = ((-2, 2), (6, 4))
    params.wall_pos = ((5, 5), (4, 5), (4, 4))
    params.max_steps = 30
    params.env_name = "example-env"
    # params.human_policy = "noop"
    env = HumanVatGoalEnv(env_params=params)

    # env = BasicVatGoalEnv()

    print("observation_dim: ", env.observation_dim)
    print("action_space: ", env.action_space)
    print("observation_space: ", env.observation_space)
    print("descr: ", env.descr)

    # env.render()
    # time.sleep(8.0)
    # env.close()

    for i_episode in range(5):
        obs = env.reset()
        for t in range(100):
            env.render(mode="window")
            time.sleep(0.1)

            action = env.action_space.sample()
            print(
                "step=%2d\t" % (env.step_count),
                env._decode(obs[0]),
                "->",
                env.actions(action),
                end="",
            )

            obs, reward, done, info = env.step(action)
            print("\treward=%.2f" % (reward))

            if done:
                env.render(mode="window")
                print("done!")
                print(info)
                print("-" * 20)
                time.sleep(0.2)
                break

    env.close()
