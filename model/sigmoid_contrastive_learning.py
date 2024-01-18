import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
import torch.distributed as dist
from musiclm_pytorch.distributed import AllGather
from einops import rearrange
from model_utils import exists


class SigmoidContrastiveLearning(nn.Module):

    def __init__(
            self,
            *,
            layers=1,
            init_temp=10,
            init_bias=-10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)
        self.all_gather = AllGather(dim=1, all_reduce_grads=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        device = self.device

        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        text_latents, rank_sizes = self.all_gather(text_latents)
        n = text_latents.shape[1]
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)
        sims = sims * self.temperatures.exp() + self.bias
        labels = torch.eye(n, device=device)

        if exists(rank_sizes):
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim=0)
            labels = labels_by_ranks[dist.get_rank()]

        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)
        return -F.logsigmoid(labels * sims).sum() / n
