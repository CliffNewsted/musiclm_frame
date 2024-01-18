import torch
import torch.nn.functional as F
from torch import nn
from x_clip.tokenizer import tokenizer
from einops import repeat, pack, unpack
from beartype.typing import List, Optional, Tuple
from beartype import beartype
from transformer import LayerNorm, Transformer
from model_utils import exists

# text transformer
class TextTransformer(nn.Module):
    @beartype
    def __init__(
            self,
            dim,
            depth,
            num_tokens=tokenizer.vocab_size,
            max_seq_len=256,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            pad_id=0
    ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult
        )

        self.pad_id = pad_id
        self.norm = LayerNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
            self,
            x=None,
            raw_texts: Optional[List[str]] = None,
            mask=None,
            return_all_layers=False
    ):
        assert exists(x) ^ exists(raw_texts)
        if exists(raw_texts):
            x = tokenizer.tokenize(raw_texts).to(self.device)

        if not exists(mask):
            mask = x != self.pad_id

        b, n, device = *x.shape, x.device
        # token embedding + positional embedding

        x = self.token_emb(x)
        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'
        x = x + self.pos_emb(torch.arange(n, device=device))

        # cls tokens, as in bert
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x, ps = pack([cls_tokens, x], 'b * d')
        # account for attending to cls token with self attention mask

        mask = F.pad(mask, (1, 0), value=True)
        # attention
        x, all_layers = self.transformer(x, mask=mask, return_all_layers=True)

        # unpack the cls tokens
        cls_tokens, _ = unpack(x, ps, 'b * d')
        out = self.norm(cls_tokens)

        if not return_all_layers:
            return out

        return out, all_layers