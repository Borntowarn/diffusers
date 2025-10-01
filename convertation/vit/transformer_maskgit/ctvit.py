from torch import nn

from einops import rearrange, pack
from einops.layers.torch import Rearrange

from .vector_quantize_pytorch import VectorQuantize

from .attention import Transformer, ContinuousPositionBias

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


class CTViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        spatial_depth,
        temporal_depth,
        discr_base_dim = 16,
        dim_head = 64,
        heads = 8,
        channels = 1,
        use_vgg_and_gan = True,
        vgg = None,
        discr_attn_res_layers = (16,),
        use_hinge_loss = True,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )

        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim = dim, codebook_size = codebook_size, use_cosine_sim = True)



    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tokens.shape[:-1]

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # encode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        return tokens

    def forward(
        self,
        video,
        mask = None,
        return_recons = False,
        return_recons_only = False,
        return_discr_loss = False,
        apply_grad_penalty = True,
        return_only_codebook_ids = False,
        return_encoded_tokens=False
    ):
        tokens = self.to_patch_emb(video)

        shape = tokens.shape
        *_, h, w, _ = shape

        tokens = self.encode(tokens)

        tokens, packed_fhw_shape = pack([tokens], 'b * d')

        tokens, indices = self.vq(tokens, mask = None)

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        return tokens