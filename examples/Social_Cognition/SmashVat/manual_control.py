import time
from window import Window
from environment import *


class ManualControl(object):
    """ManualControl of HumanVatGoalEnv Class"""

    def __init__(self, env=HumanVatGoalEnv()):
        self.env = env
        self.window = Window(env.descr + "[manual]")
        self.window.reg_key_press_handler(self._key_handler)

    def display(self):
        self._reset()

        # Blocking event loop
        self.window.show(block=True)
        return

    def _redraw(self):
        img = self.env.render("rgb_array", cell_size=64)
        self.window.show_img(img)
        return

    def _reset(self):
        self.env.reset()
        print("-" * 20)
        print(
            "step=%2d " % (self.env.step_count),
            "obs=",
            [self.env._decode(o) for o in self.env._gen_obs()],
            end=" -> ",
            flush=True,
        )
        self._redraw()
        return

    def _step(self, action):

        obs, reward, done, info = self.env.step(action)
        print(self.env.actions(action), "\treward=%.2f" % (reward))
        print(
            "step=%2d " % (self.env.step_count),
            "obs=",
            [self.env._decode(o) for o in self.env._gen_obs()],
            end=" -> ",
            flush=True,
        )

        self._redraw()
        if done:
            print("done!")
            print(info)
            time.sleep(0.2)
            self._reset()
        return

    def _key_handler(self, event):
        # print('pressed', event.key)

        if event.key == "escape" or event.key == "q":
            self.window.close()
        elif event.key == "backspace":
            self._reset()
        elif event.key == "left":
            self._step(self.env.actions.left)
        elif event.key == "right":
            self._step(self.env.actions.right)
        elif event.key == "up":
            self._step(self.env.actions.up)
        elif event.key == "down":
            self._step(self.env.actions.down)
        elif event.key == " ":  # Spacebar
            self._step(self.env.actions.noop)
        elif event.key == "enter":  # Smash
            self._step(self.env.actions.smash)

        return


if __name__ == "__main__":

    mc = ManualControl(env=HumanVatGoalEnv())
    mc.display()

