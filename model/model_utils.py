from functools import wraps
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


def exists(val):
    return val is not None


def first(it):
    return it[0]


def default(val, d):
    return val if exists(val) else d


def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def pad_dim_to(t, length, dim=0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


# tensor functions
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)


# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned
def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    (_, h, w, dim), device, dtype = *patches.shape, patches.device, patches.dtype
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    pe = pe.type(dtype)
    return rearrange(pe, '(h w) d -> h w d', h=h, w=w)


# hierarchical cl loss
def interspersed_indices(layers, total_layers):
    assert total_layers >= layers
    step = total_layers / layers
    return (torch.arange(0, layers) * step).floor().long()


def pair(t):
    return (t, t) if not isinstance(t, tuple) else t
