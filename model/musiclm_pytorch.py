import torch
from torch import nn
from audiolm_pytorch import AudioLM
from x_clip.tokenizer import tokenizer
from beartype import beartype
from mulan_pytorch import MuLaNEmbedQuantizer
from model_utils import exists, first


class MusicLM(nn.Module):
    @beartype
    def __init__(self, audio_lm: AudioLM, mulan_embed_quantizer: MuLaNEmbedQuantizer()):
        super().__init__()
        assert not exists(audio_lm.audio_conditioner)
        self.mulan_embed_quantizer = mulan_embed_quantizer
        self.audio_lm = audio_lm

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, text: str, num_samples=1, **audio_lm_kwargs):
        self.eval()
        texts = tokenizer.tokenize([text]).to(self.device)
        text_embeds = self.mulan_embed_quantizer(texts=texts)

        # unable to deal with variable lengthed audio for now
        samples = []
        for _ in range(num_samples):
            music = self.audio_lm(text_embeds=text_embeds, **audio_lm_kwargs)
            samples.append(music)

        # if one sample, just return it
        if num_samples == 1:
            return first(samples)
        mulan = self.mulan_embed_quantizer.mulan

        # get the one with the highest similarity score, of all the samples
        sims = torch.cat([mulan(texts=texts, wavs=music, return_similarities=True) for music in samples], dim=0)
        top_matching_index = sims.topk(1, dim=0).indices.item()
        return samples[top_matching_index]
