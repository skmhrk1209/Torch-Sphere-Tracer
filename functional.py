import torch
import torch.nn as nn
import numpy as np


# ---------------- GLSL-like helper functions ---------------- #

def length(x):
    return torch.norm(x, dim=-1, keepdim=True)


def maximum(x, y, *z):
    y = y if isinstance(y, torch.Tensor) else full(x, y)
    m = torch.max(x, y)
    return maximum(m, *z) if z else m


def minimum(x, y, *z):
    y = y if isinstance(y, torch.Tensor) else full(x, y)
    m = torch.min(x, y)
    return minimum(m, *z) if z else m


def clamp(x, a, b):
    return minimum(maximum(x, a), b)


def mix(x, y, a):
    return x * (1.0 - a) + y * a


def stack(*x):
    return torch.stack(x, dim=-2)


def unstack(x):
    return torch.unbind(x, dim=-2)


def concat(*x):
    return torch.cat(x, dim=-1)


def unconcat(x):
    return torch.unbind(unsqueeze(x), dim=-2)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def unsqueeze(x):
    return torch.unsqueeze(x, dim=-1)


def matmul(A, x):
    return squeeze(A @ unsqueeze(x))


def dot(x, y):
    return torch.sum(x * y, dim=-1, keepdim=True)


def ndot(x, y):
    x, y = unconcat(x * y)
    return x - y


def transpose(A):
    return torch.transpose(A, -2, -1)


def abs(x):
    return torch.abs(x)


def sqrt(x):
    return torch.sqrt(x + 1e-6)


def sign(x):
    return torch.sign(x)


def cos(x):
    return torch.cos(x)


def sin(x):
    return torch.sin(x)


def relu(x):
    return nn.functional.relu(x)


def mod(x, y):
    return torch.fmod(x, y)


def round(x):
    return torch.round(x)


def zero(x):
    return torch.zeros_like(x)


def one(x):
    return torch.ones_like(x)


def full(x, y):
    return torch.full_like(x, y)


def expand(x, y):
    while x.dim() < y.dim():
        x = torch.unsqueeze(x, dim=0)
    x = x.expand_as(y)
    return x


def tensor(x, y):
    return x.new_tensor(y)
