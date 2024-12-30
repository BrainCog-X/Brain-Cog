# Code modified from:
# https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/window.py

import sys
import numpy as np
import matplotlib.pyplot as plt


class Window(object):
    """Interactive Window for Image Display, using matplotlib"""

    def __init__(self, title):

        self.fig, self.ax = plt.subplots()
        self.ax.axis("off")  # clear x-axis and y-axis

        self.title = title
        self.set_window_title(self.title)

        self.key_press_handler = self._default_key_press_handler
        self.reg_key_press_handler(self.key_press_handler)

        self.img_shown = None

        return

    def set_window_title(self, title):
        self.title = title
        # https://stackoverflow.com/questions/5812960/change-figure-window-title-in-pylab
        self.fig.canvas.manager.set_window_title(self.title)
        return

    def reg_key_press_handler(self, key_press_handler):
        self.key_press_handler = key_press_handler
        self.fig.canvas.mpl_connect("key_press_event", self.key_press_handler)
        return

    def _default_key_press_handler(self, event):
        print("press", event.key)
        sys.stdout.flush()
        if event.key == "escape":
            self.close()

    def show(self, block=True):

        if not self.is_open():

            # https://stackoverflow.com/questions/31729948/matplotlib-how-to-show-a-figure-that-has-been-closed
            # if window has been closed by plt.close()
            # create a dummy figure and use its manager to display "fig"
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = self.fig
            self.fig.set_canvas(new_manager.canvas)

            self.set_window_title(self.title)
            self.reg_key_press_handler(self.key_press_handler)

        if not block:
            plt.ion()
        else:
            plt.ioff()

        plt.show()

        # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
        # https://stackoverflow.com/questions/53758472/why-is-plt-pause-not-described-in-any-tutorials-if-it-is-so-essential-or-am-i
        plt.pause(0.001)

        return

    def show_img(self, img):

        if self.img_shown == None:
            self.img_shown = self.ax.imshow(img)
        else:
            self.img_shown.set_data(img)

        self.fig.canvas.draw()

        # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
        # https://stackoverflow.com/questions/53758472/why-is-plt-pause-not-described-in-any-tutorials-if-it-is-so-essential-or-am-i
        plt.pause(0.001)

        return

    def close(self):
        plt.close(self.fig)
        return

    def is_open(self):
        # https://stackoverflow.com/questions/7557098/matplotlib-interactive-mode-determine-if-figure-window-is-still-displayed
        return bool(plt.get_fignums())


if __name__ == "__main__":

    window = Window("TestWindow")

    def on_press(event):
        print("press", event.key)
        sys.stdout.flush()
        if event.key == "escape":
            window.close()
        elif event.key == "x":
            img = np.full(shape=(7 * 32, 5 * 32, 3), fill_value=55).astype(np.uint8)
            window.show_img(img)
        elif event.key == "c":
            img = np.full(shape=(7 * 32, 5 * 32, 3), fill_value=155).astype(np.uint8)
            window.show_img(img)

    window.reg_key_press_handler(on_press)

    print(window.is_open())  # True

    window.show(block=True)
    print(window.is_open())  # False

    window.show(block=False)
    print(window.is_open())  # True

    plt.pause(2.0)
    window.close()
    print(window.is_open())  # False

    window.show(block=True)

