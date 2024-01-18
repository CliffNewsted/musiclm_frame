import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from model_utils import exists, default, l2norm


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.):
    dim_hidden = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias=False)
    )


class LayerNorm(nn.Module):
    def __init__(self, dim, scale=True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        self.register_buffer('gamma', torch.ones(dim), persistent=False)
        self.register_buffer('beta', torch.zeros(dim), persistent=False)

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)


# attention
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            causal=False,
            dim_head=64,
            heads=8,
            dropout=0.,
            scale=8
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, rel_pos_bias=None, mask=None):
        (b, n, _), device = *x.shape, x.device
        x = self.norm(x)

        # project for queries, keys, values
        q, (k, v) = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)

        # split for multi-headed attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, dim_head=64, heads=8, attn_dropout=0., ff_mult=4, ff_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

    def forward(self, x, rel_pos_bias=None, mask=None, return_all_layers=False):
        layers = []
        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias=rel_pos_bias, mask=mask) + x
            x = ff(x) + x
            layers.append(x)

        if not return_all_layers:
            return x

        return x, torch.stack(layers[:-1])
