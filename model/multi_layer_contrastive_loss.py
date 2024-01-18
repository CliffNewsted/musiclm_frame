from functools import partial
import torch
from torch import nn, einsum
from einops import reduce
from transformer import LayerNorm
from sigmoid_contrastive_learning import SigmoidContrastiveLearning
from softmax_contrastive_learning import SoftmaxContrastiveLearning
from model_utils import l2norm


class MultiLayerContrastiveLoss(nn.Module):
    def __init__(
            self,
            *,
            audio_dim,
            text_dim,
            dim_latent,
            layers,
            decoupled_contrastive_learning=False,
            sigmoid_contrastive_loss=False
    ):
        super().__init__()
        self.layers = layers

        self.audio_norm = LayerNorm(audio_dim, scale=False)
        self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))
        self.audio_latent_weight = nn.Parameter(torch.randn(layers, audio_dim, dim_latent))
        self.audio_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        self.text_norm = LayerNorm(text_dim, scale=False)
        self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))
        self.text_latent_weight = nn.Parameter(torch.randn(layers, text_dim, dim_latent))
        self.text_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning,
                                                                                    decoupled_contrastive_learning=decoupled_contrastive_learning)
        self.contrast = klass(layers=layers)

    def forward(self, *, audio_layers, text_layers):
        device, batch = audio_layers.device, audio_layers.shape[1]

        audio_gap = reduce(audio_layers, 'l b n d -> l b d', 'mean')
        audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma
        audio_latents = einsum('l b d, l d e -> l b e', audio_embeds, self.audio_latent_weight) + self.audio_latent_bias
        audio_latents = l2norm(audio_latents)

        text_cls_tokens = text_layers[:, :, 0]
        text_embeds = self.text_norm(text_cls_tokens) * self.text_gamma
        text_latents = einsum('l b d, l d e -> l b e', text_embeds, self.text_latent_weight) + self.text_latent_bias
        text_latents = l2norm(text_latents)

        return self.contrast(audio_latents, text_latents)
