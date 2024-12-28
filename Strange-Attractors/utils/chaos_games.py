import torch
import numpy
from utils.utils import *


def jump_game(n, r, device):

    sampler = torch.distributions.categorical.Categorical(
        probs=torch.ones(n, device=device))

    ptsx = cos(torch.linspace(0, 2*np.pi*(n-1)/n, n))
    ptsy = sin(torch.linspace(0, 2*np.pi*(n-1)/n, n))

    def game(pos):

        x, y = unconcat(pos)
        sample = sampler.sample(x.shape)

        x1 = torch.zeros_like(x)
        y1 = torch.zeros_like(y)

        for i in range(n):

            x1 = torch.where(sample == i, (ptsx[i]-x)*r + x, x1)
            y1 = torch.where(sample == i, (ptsy[i]-y)*r + y, y1)

        return concat(x1, y1)

    return game
