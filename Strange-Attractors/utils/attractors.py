import torch
from utils.utils import *

def clifford_attractor(a, b, c, d):

    def attractor(pos):
        x, y = unconcat(pos)
        x1 = sin(a*y) + c*cos(a*x)
        y1 = sin(b*x) + d*cos(b*y)

        return concat(x1, y1)

    return attractor


def ring_attractor(a, b, c, d):

    def attractor(pos):
        x, y = unconcat(pos)
        x1 = d*sin(a*y) - sin(b*x)
        y1 = c*cos(a*y) + cos(b*x)

        return concat(x1, y1)

    return attractor