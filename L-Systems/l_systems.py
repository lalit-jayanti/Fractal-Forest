import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import time


def calc_time(func):
    def inner(*args, **kwargs):
        begin = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} {end-begin} seconds")
        return ret
    return inner


class LsystemParser():

    def __init__(self, V, w, P):
        self.V = V
        self.w = w
        self.P = P

        self.tmp_w = ""

        self.pos = np.array([0.0, 0.0, 0.0])  # x, y, angle
        self.pos_stack = []

        self.special_chars = ['[', ']']
        self.V.update({self.special_chars[0]: (0, 0),
                       self.special_chars[1]: (0, 0)})

        self.color = [0.215, 0.776, 1.00]
        self.linewidth = 0.5

        self.move_chars = 0  # for generating gradient
        self.x = 0  # parameter for color generation

        # It is implicitly assumed if a rule is
        # not given for an Alphabet, then it is a constant
        # update P with constants
        new_keys = []
        for i in self.P:
            for j in self.P[i]:
                if j not in self.P:
                    if j not in new_keys:
                        new_keys.append(j)
        for j in self.w:
            if j not in new_keys:
                if j not in self.P:
                    new_keys.append(j)

        for k in new_keys:
            self.P.update({k: k})

    ## Methods to generate strings / representation of fractals ##

    def generate_string_util(self, w, order):
        tmp_w = ""

        if (order == 0):
            return w

        for i in w:
            tmp_w += self.P[i]

        w = self.generate_string_util(tmp_w, order-1)
        return w

    @calc_time
    def generate_string(self, order):
        self.tmp_w = self.generate_string_util(self.w, order)
        return self.tmp_w

    ## Methods to decode / visualize strings ##

    def move(self, i):
        end = np.array([0.0, 0.0, 0.0])
        end[2] = self.pos[2] + self.V[i][1]
        end[0] = self.pos[0] + self.V[i][0]*np.cos(end[2]*np.pi/180)
        end[1] = self.pos[1] + self.V[i][0]*np.sin(end[2]*np.pi/180)
        return end

    def count_move_chars(self):
        for i in self.tmp_w:
            if not (self.V[i][0] == 0):
                self.move_chars += 1

    def color_update(self, g):
        if g == 0:
            self.color = [min(1.0, self.x),
                          min(1.0, self.x),
                          min(1.0, self.x)]
        if g == 1:
            if (self.x < 0.33):
                self.color = [1.0-2.99*self.x,
                              2.99*self.x,
                              0.0]
            elif (self.x < 0.66):
                self.color = [0.0,
                              1.0-2.99*(self.x-0.33),
                              2.99*(self.x-0.33)]
            else:
                self.color = [2.99*(self.x-0.66),
                              0.0,
                              1.0-2.99*(self.x-0.66)]

            self.color[0] = max(0.0, min(self.color[0], 1.0))
            self.color[1] = max(0.0, min(self.color[1], 1.0))
            self.color[2] = max(0.0, min(self.color[2], 1.0))

        if g == 2:
            self.color = hsv_to_rgb([self.x, 1.0, 1.0])

    @calc_time
    def visualize(self, gradient=None):
        if gradient is not None:
            self.count_move_chars()
            self.color = [0.0, 0.0, 0.0]

        ax = plt.axes()
        ax.set_facecolor("black")
        ax.set_aspect('equal')
        ax.axis('off')
        for i in self.tmp_w:

            # handle special chars '[', ']'
            if i == self.special_chars[0]:
                self.pos_stack.append(self.pos)
                continue
            if i == self.special_chars[1]:
                if len(self.pos_stack) != 0:
                    self.pos = self.pos_stack.pop()
                    continue

            # for ordinary chars
            end = self.move(i)
            if not (end[0] == self.pos[0] and end[1] == self.pos[1]):  # ignore redundant moves
                plt.plot([self.pos[0], end[0]],
                         [self.pos[1], end[1]],
                         color=[self.color[0], self.color[1], self.color[2]],
                         linewidth=self.linewidth)

                if gradient is not None:
                    self.x += 1/self.move_chars
                    self.color_update(gradient)

            self.pos = end
