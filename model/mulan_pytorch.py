from functools import partial
import torch
from torch import nn, einsum
from audiolm_pytorch.utils import AudioConditionerBase
from vector_quantize_pytorch import ResidualVQ
from einops import rearrange, repeat
from beartype.typing import List, Optional, Tuple
from beartype import beartype
from transformer_audio import AudioSpectrogramTransformer
from transformer_text import TextTransformer
from sigmoid_contrastive_learning import SigmoidContrastiveLearning
from softmax_contrastive_learning import SoftmaxContrastiveLearning
from multi_layer_contrastive_loss import MultiLayerContrastiveLoss
from model_utils import default, interspersed_indices, l2norm, exists


class MuLaN(nn.Module):
    @beartype
    def __init__(
            self,
            audio_transformer: AudioSpectrogramTransformer,
            text_transformer: TextTransformer,
            dim_latent=128,  # they use 128
            decoupled_contrastive_learning=True,  # think this was used, make it optional
            hierarchical_contrastive_loss=False,
            hierarchical_contrastive_loss_layers=None,
            sigmoid_contrastive_loss=False
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.audio = audio_transformer
        self.text = text_transformer
        self.text_to_latents = nn.Linear(self.text.dim, dim_latent)
        self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)
        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning,
                                                                                    decoupled_contrastive_learning=decoupled_contrastive_learning)
        self.contrast = klass()
        self.multi_layer_contrastive_learning = None

        if hierarchical_contrastive_loss:
            num_layers = default(hierarchical_contrastive_loss_layers,
                                 min(audio_transformer.depth, text_transformer.depth) - 1)
            assert num_layers > 0

            self.register_buffer('text_layers_indices', interspersed_indices(num_layers, text_transformer.depth))
            self.register_buffer('audio_layers_indices', interspersed_indices(num_layers, audio_transformer.depth))

            self.multi_layer_contrastive_learning = MultiLayerContrastiveLoss(
                audio_dim=self.audio.dim,
                text_dim=self.text.dim,
                dim_latent=dim_latent,
                layers=num_layers,
                decoupled_contrastive_learning=decoupled_contrastive_learning,
                sigmoid_contrastive_loss=sigmoid_contrastive_loss
            )

    def get_audio_latents(self, wavs, return_all_layers=False):
        audio_embeds, audio_layers = self.audio(wavs, return_all_layers=True)
        audio_latents = self.audio_to_latents(audio_embeds)
        out = l2norm(audio_latents)
        if not return_all_layers:
            return out
        return out, audio_layers

    @beartype
    def get_text_latents(
            self,
            texts=None,
            raw_texts: Optional[List[str]] = None,
            return_all_layers=False
    ):
        text_embeds, text_layers = self.text(texts, raw_texts=raw_texts, return_all_layers=True)
        text_latents = self.text_to_latents(text_embeds)
        out = l2norm(text_latents)
        if not return_all_layers:
            return out

        return out, text_layers

    @beartype
    def forward(
            self,
            wavs,
            texts=None,
            raw_texts: Optional[List[str]] = None,
            return_latents=False,
            return_similarities=False,
            return_pairwise_similarities=False
    ):
        batch, device = wavs.shape[0], wavs.device
        audio_latents, audio_layers = self.get_audio_latents(wavs, return_all_layers=True)
        text_latents, text_layers = self.get_text_latents(texts, raw_texts=raw_texts, return_all_layers=True)

        if return_latents:
            return audio_latents, text_latents

        if return_similarities:
            return einsum('i d, i d -> i', audio_latents, text_latents)

        if return_pairwise_similarities:
            cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents)
            return cosine_sim

        cl_loss = self.contrast(audio_latents, text_latents)

        if not exists(self.multi_layer_contrastive_learning):
            return cl_loss

        audio_layers = audio_layers[self.audio_layers_indices]
        text_layers = text_layers[self.text_layers_indices]
        hierarchical_cl_loss = self.multi_layer_contrastive_learning(
            audio_layers=audio_layers,
            text_layers=text_layers
        )
        return cl_loss + hierarchical_cl_loss


class MuLaNEmbedQuantizer(AudioConditionerBase):
    @beartype
    def __init__(
            self,
            mulan: MuLaN,
            conditioning_dims: Tuple[int, ...],
            rq_num_quantizers=8,
            rq_ema_decay=0.9,
            codebook_size=1024,
            namespaces: Tuple[str, ...] = ('semantic', 'coarse', 'fine'),

    ):
        super().__init__()
        self.mulan = mulan
        assert len(namespaces) > 0
        self.namespaces = namespaces
        self.conditioning_dims = conditioning_dims

        assert len(conditioning_dims) == len(
            namespaces), 'number of conditioning dimensions must be equal to number of namespaces'

        dim = mulan.dim_latent

        self.rq = ResidualVQ(
            dim=dim,
            num_quantizers=rq_num_quantizers,
            codebook_size=codebook_size,
            decay=rq_ema_decay,
            commitment_weight=0,  # only use EMA to update codebooks
            kmeans_init=True,
            threshold_ema_dead_code=2,
            quantize_dropout=False  # no quantize dropout
        )

        self.dim = dim
        self.num_codebooks = rq_num_quantizers
        self.cond_embeddings = nn.ParameterDict({})

        for namespace, conditioning_dim in zip(namespaces, conditioning_dims):
            cond_embeddings = nn.Parameter(torch.randn(rq_num_quantizers, codebook_size, conditioning_dim))
            nn.init.normal_(cond_embeddings, std=0.02)
            self.cond_embeddings[namespace] = cond_embeddings
        self.set_default_namespace(namespaces[0])

    def parameters(self):
        return self.cond_embeddings.parameters()

    def set_default_namespace(self, namespace):
        self._default_namespace = namespace

    def forward(self, wavs=None, texts=None, namespace=None):
        assert exists(wavs) ^ exists(texts)
        namespace = default(namespace, self._default_namespace)
        assert namespace in self.namespaces, f'namespace {namespace} not found'
        cond_embeddings = self.cond_embeddings[namespace]

        with torch.no_grad():
            self.mulan.eval()
            if exists(wavs):
                latents = self.mulan.get_audio_latents(wavs)
            elif exists(texts):
                latents = self.mulan.get_text_latents(texts)

        _, indices, _ = self.rq(latents)
        batch, num_codebooks, dim = indices.shape[0], self.num_codebooks, cond_embeddings.shape[-1]
        cond_embeddings = repeat(cond_embeddings, 'q c d -> b q c d', b=batch)
        indices = repeat(indices, 'b q -> b q 1 d', q=num_codebooks, d=dim)
        cond_embeddings = cond_embeddings.gather(2, indices)
        return rearrange(cond_embeddings, 'b q 1 d -> b q d')
