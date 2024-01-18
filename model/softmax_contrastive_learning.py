import math
import torch
from torch import nn, einsum
from musiclm_pytorch.distributed import AllGather
from einops import rearrange, reduce
from model_utils import log, matrix_diag


# contrastive losses
class SoftmaxContrastiveLearning(nn.Module):
    def __init__(self, *, layers=1, decoupled_contrastive_learning=False, init_temp=10):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.decoupled_contrastive_learning = decoupled_contrastive_learning
        self.all_gather = AllGather(dim=2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')
        batch = audio_latents.shape[1]

        if self.all_gather.is_distributed:
            latents = torch.stack((audio_latents, text_latents))
            latents, _ = self.all_gather(latents)
            audio_latents, text_latents = latents

        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)
        sims = sims * self.temperatures.exp()
        cosine_sims_exp = sims.exp()
        numerator = matrix_diag(cosine_sims_exp)

        if self.decoupled_contrastive_learning:
            eye = torch.eye(batch, device=self.device, dtype=torch.bool)
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.)

        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')
        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))
        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum()
