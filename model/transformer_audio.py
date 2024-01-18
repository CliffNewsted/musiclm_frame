import torch
from torch import nn
from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from transformer import LayerNorm, Transformer
from model_utils import pair, round_down_nearest_multiple, print_once, posemb_sincos_2d, Sequential


class AudioSpectrogramTransformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            patch_size=16,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_mult=4,
            ff_dropout=0.,
            accept_spec=False,
            accept_spec_time_first=True,
            spec_n_fft=128,
            spec_power=2,
            spec_win_length=24,
            spec_hop_length=None,
            spec_pad=0,
            spec_center=True,
            spec_pad_mode='reflect',
            spec_aug_stretch_factor=0.8,
            spec_aug_freq_mask=80,
            spec_aug_time_mask=80,
            patch_dropout_prob=0.25
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.patch_size = pair(patch_size)
        patch_input_dim = self.patch_size[0] * self.patch_size[1]

        self.to_patch_tokens = Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1=self.patch_size[0], p2=self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        self.accept_spec = accept_spec
        self.accept_spec_time_first = accept_spec_time_first

        self.spec = Spectrogram(
            n_fft=spec_n_fft,
            power=spec_power,
            win_length=spec_win_length,
            hop_length=spec_hop_length,
            pad=spec_pad,
            center=spec_center,
            pad_mode=spec_pad_mode
        )

        # SpecAugment - seems to be widely used in audio field https://arxiv.org/abs/1904.08779

        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=spec_aug_freq_mask),
            TimeMasking(time_mask_param=spec_aug_time_mask),
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout
        )

        self.norm = LayerNorm(dim)

        # patch dropout
        self.patch_dropout_prob = patch_dropout_prob

        # 2d dynamic positional bias
        mlp_hidden_dim = dim // 4
        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
            self,
            x,
            force_no_patch_dropout=False,
            return_all_layers=False
    ):
        batch, device = x.shape[0], x.device
        assert (self.accept_spec and x.ndim == 3) or (not self.accept_spec and x.ndim == 2)

        if self.accept_spec and self.accept_spec_time_first:
            x = rearrange(x, 'b t f -> b f t')

        if not self.accept_spec:
            x = self.spec(x)

        if self.training:
            x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes
        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size
        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args),
                                            ((height, patch_height), (width, patch_width)))
        if (height, width) != (rounded_height, rounded_width):  # just keep printing to be annoying until it is fixed
            print_once(
                f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., :rounded_height, :rounded_width]
        # to patches
        x = self.to_patch_tokens(x)

        # get number of patches along height and width
        _, num_patch_height, num_patch_width, _ = x.shape

        # get 2d relative positions
        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device=device),
            torch.arange(num_patch_width, device=device)
            , indexing='ij'), dim=-1)

        grid = rearrange(grid, '... c -> (...) c')

        # 2d sinusoidal positional embedding
        x = x + posemb_sincos_2d(x)
        x = rearrange(x, 'b ... c -> b (...) c')

        # patch dropout
        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device
            batch_indices = torch.arange(batch, device=device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device=device).topk(num_patches_keep, dim=-1).indices
            x = x[batch_indices, patch_indices_keep]
            grid = repeat(grid, '... -> b ...', b=batch)
            grid = grid[batch_indices, patch_indices_keep]

        # 2d relative positional bias
        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())
        x, all_layers = self.transformer(x, rel_pos_bias=rel_pos_bias, return_all_layers=True)
        x = reduce(x, 'b n d -> b d', 'mean')
        out = self.norm(x)

        if not return_all_layers:
            return out

        return out, all_layers
