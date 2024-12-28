import torch
import numpy as np


def render(
    num_points,
    attractor,
    image_height,
    image_width,
    sx,sy,rz,
    initial_iterations,
    max_iterations,
    device='cuda'):

    device = 'cuda'
    pos = torch.randn((num_points, 2), device=device)
    at = attractor

    h = image_height
    w = image_width

    img = torch.zeros((h, w), device=device)

    def pos_to_img(pos): 
        
        return transform(pos=pos.clone().detach(),
                         sx=sx, sy=sy,
                         rz=rz,
                         dx=h/2, dy=w/2)
        
    for i in range(initial_iterations):
        pos = at(pos)
    
    for i in range(max_iterations):
    
        pos = at(pos)
        img_idx = pos_to_img(pos).type(dtype=torch.int32)
        xi, yi = unconcat(img_idx)
        xi = minimum(maximum(xi, 0), h-1)
        yi = minimum(maximum(yi, 0), w-1)
        img[xi, yi] += 1
    
    img = img.detach().to('cpu').numpy()

    return img


    
def transform(pos,
              sx=1.0, sy=1.0,
              rz=0.0,
              dx=0.0, dy=0.0,
              ox=0.0, oy=0.0):

    device = pos.device

    ons = torch.ones_like(length(pos))
    x, y = unconcat(pos)
    pos = concat(x, y, ons)

    otf = torch.tensor([[1.0, 0.0, ox],
                        [0.0, 1.0, oy],
                        [0.0, 0.0, 1.0]],
                       device=device, dtype=torch.float32)

    stf = torch.tensor([[sx,  0.0, 0.0],
                        [0.0,  sy, 0.0],
                        [0.0, 0.0, 1.0]],
                       device=device, dtype=torch.float32)

    rtf = torch.tensor([[np.cos(rz),  -np.sin(rz), 0.0],
                        [np.sin(rz),   np.cos(rz), 0.0],
                        [0.0,          0.0,        1.0]],
                       device=device, dtype=torch.float32)

    otf_inv = torch.tensor([[1.0, 0.0,  -ox],
                            [0.0, 1.0,  -oy],
                            [0.0, 0.0, 1.0]],
                           device=device, dtype=torch.float32)

    dtf = torch.tensor([[1.0, 0.0,  dx],
                        [0.0, 1.0,  dy],
                        [0.0, 0.0, 1.0]],
                       device=device, dtype=torch.float32)

    tf = dtf@otf_inv@rtf@stf@otf
    pos = matmul(tf, pos)
    x, y, ons = unconcat(pos)
    return concat(x, y)


def manhattan_distance(x):
    x1, y1 = unconcat(x)
    return torch.abs(x1)+torch.abs(y1)


def chebyshev_distance(x):
    x1, y1 = unconcat(x)
    return maximum(torch.abs(x1), torch.abs(y1))


def eucledian_distance(x):
    return torch.norm(x, dim=-1, keepdim=True)


def length(x):
    return eucledian_distance(x)
    # return chebyshev_distance(x)
    # return manhattan_distance(x)


def maximum(x, y, *z):
    y = y if isinstance(y, torch.Tensor) else torch.full_like(x, y)
    m = torch.max(x, y)
    return maximum(m, *z) if z else m


def minimum(x, y, *z):
    y = y if isinstance(y, torch.Tensor) else torch.full_like(x, y)
    m = torch.min(x, y)
    return minimum(m, *z) if z else m


def concat(*x):
    return torch.cat(x, dim=-1)


def unconcat(x):
    return torch.unbind(unsqueeze(x), dim=-2)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def unsqueeze(x):
    return torch.unsqueeze(x, dim=-1)


def matmul(A, x):
    return squeeze(A@unsqueeze(x))


def cpow(x, p):

    x1, y1 = unconcat(x)
    tht = atan2(y1, x1)*p
    rad = length(x)**p
    res = concat(rad*cos(tht), rad*sin(tht))
    return res


def cmul(a, b):

    x1, y1 = unconcat(a)
    x2, y2 = unconcat(b)

    res = concat(x1*x2 - y1*y2, x1*y2 + x2*y1)
    return res


def cos(x):
    return torch.cos(x)


def sin(x):
    return torch.sin(x)


def log(x):
    return torch.log(x)


def atan2(y, x):
    return torch.arctan2(y, x)
