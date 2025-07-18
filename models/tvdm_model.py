# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
#from torch.utils.checkpoint import checkpoint
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    #print(">>> modulate shapes:", x.shape, shift.shape, scale.shape)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    # x: (B, T, D)
    # shift, scale: (B, D) → need to expand to (B, T, D)
    '''
    B, T, D = x.shape
    sB = shift.shape[0]

    if sB != B:
        if B % sB != 0:
            raise RuntimeError(f"Incompatible shape: x={x.shape}, shift={shift.shape}")
        reps = B // sB
        shift = shift.repeat_interleave(reps, dim=0)
        scale = scale.repeat_interleave(reps, dim=0)

    shift = shift[:, None, :].expand(B, T, D)
    scale = scale[:, None, :].expand(B, T, D)
    return x * (1 + scale) + shift
    '''

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # General tvdm training
        ###########################
        #       Lora config       #
        '''
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        '''

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        '''
        q = self.to_q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = self.to_k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        v = self.to_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        '''
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            #print("using xformers!")
            x = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These  be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Tvdm Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A tvdm block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of tvdm.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Tvdm(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        extras=2,
        attention_mode='math',
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.extras == 2:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1024, hidden_size, bias=True)
        )
        
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in tvdm blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, x, t, y=None, use_fp16=False, y_image=None, use_image_num=0):
        """
        Forward pass of tvdm.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        y_image: tensor of video frames
        use_image_num: how many video frames are used
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)
        batches, frames, channels, high, weight = x.shape 
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed  
        t = self.t_embedder(t, use_fp16=use_fp16)              
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.temp_embed.shape[1] + use_image_num)
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1]) 
        
        if self.extras == 2:
            y = self.y_embedder(y, self.training)
            if self.training:
                y_image_emb = []
                # print(y_image)
                for y_image_single in y_image:
                    # print(y_image_single)
                    y_image_single = y_image_single.reshape(1, -1)
                    y_image_emb.append(self.y_embedder(y_image_single, self.training))
                y_image_emb = torch.cat(y_image_emb, dim=0)
                y_spatial = repeat(y, 'n d -> n c d', c=self.temp_embed.shape[1])
                y_spatial = torch.cat([y_spatial, y_image_emb], dim=1)
                y_spatial = rearrange(y_spatial, 'n c d -> (n c) d')
            else:
                y_spatial = repeat(y, 'n d -> (n c) d', c=self.temp_embed.shape[1]) 
            
            y_temp = repeat(y, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        
        
        elif self.extras == 78:
            text_embedding = self.text_embedding_projection(text_embedding)
            text_embedding_video = text_embedding[:, :1, :]
            text_embedding_image = text_embedding[:, 1:, :]
            text_embedding_video = repeat(text_embedding, 'n t d -> n (t c) d', c=self.temp_embed.shape[1])
            text_embedding_spatial = torch.cat([text_embedding_video, text_embedding_image], dim=1)
            text_embedding_spatial = rearrange(text_embedding_spatial, 'n t d -> (n t) d')
            text_embedding_temp = repeat(text_embedding_video, 'n t d -> n (t c) d', c=self.pos_embed.shape[1])
            text_embedding_temp = rearrange(text_embedding_temp, 'n t d -> (n t) d')
        
        for i in range(0, len(self.blocks), 2):
            
            '''
            spatial_block, temp_block = self.blocks[i:i+2]
            
            #Manual Gradient Checkpointing
             
            #if self.extras == 2:
            #    c = timestep_spatial + y_spatial
            
            #elif self.extras == 78:
            #    c = timestep_spatial + text_embedding_spatial
            
            #else:
            c = timestep_spatial
            x  = spatial_block(x, c)
            #def run_spatial(x):
            #   return spatial_block(x, c)

            #x = checkpoint(run_spatial, x)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            x_video = x[:, :(frames-use_image_num), :]
            x_image = x[:, (frames-use_image_num):, :]
            
            # Add Time Embedding
            if i == 0:
                x_video = x_video + self.temp_embed 

            #if self.extras == 2:
            #    c = timestep_temp + y_temp
            
            #elif self.extras == 78:
            #    c = timestep_temp + text_embedding_temp
            
            #else:
            c = timestep_temp
            #def run_temp(x_vid):
            #    return temp_block(x_vid, c)

            x_video = temp_block(x_video, c)
            #x_video = checkpoint(run_temp, x_video)
            x = torch.cat([x_video, x_image], dim=1)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)
        '''
            spatial_block, temp_block = self.blocks[i:i+2]

            if self.extras == 2:
                c = timestep_spatial + y_spatial
            elif self.extras == 78:
                c = timestep_spatial + text_embedding_spatial
            else:
                c = timestep_spatial
            x  = spatial_block(x, c)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            x_video = x[:, :(frames-use_image_num), :]
            x_image = x[:, (frames-use_image_num):, :]
            
            # Add Time Embedding
            if i == 0:
                x_video = x_video + self.temp_embed 

            if self.extras == 2:
                c = timestep_temp + y_temp
            elif self.extras == 78:
                c = timestep_temp + text_embedding_temp
            else:
                c = timestep_temp

            x_video = temp_block(x_video, c)
            x = torch.cat([x_video, x_image], dim=1)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)
        if self.extras == 2:
            c = timestep_spatial + y_spatial
        else:
            c = timestep_spatial
        x = self.final_layer(x, c)              
        x = self.unpatchify(x)                  
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        # print(x.shape)
        return x


    def forward_with_cfg(self, x, t, y, cfg_scale, use_fp16=False):
        """
        Forward pass of tvdm, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, y, use_fp16=use_fp16)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] # 2 16 4 32 32
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   tvdm Configs                                  #
#################################################################################

def tvdm_XL_2(**kwargs):
    return Tvdm(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

tvdm_img_model = {'tvdm-XL/2': tvdm_XL_2}

if __name__ == '__main__':
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_image_num = 8

    img = torch.randn(3, 16+use_image_num, 4, 32, 32).to(device)

    t = torch.tensor([1, 2, 3]).to(device)
    y = torch.tensor([1, 2, 3]).to(device)
    y_image = [torch.tensor([48, 37, 72, 63, 74, 6, 7, 8]).to(device), 
               torch.tensor([37, 72, 63, 74, 70, 1, 2, 3]).to(device), 
               torch.tensor([72, 63, 74, 70, 71, 5, 8, 7]).to(device), 
              ]


    network = tvdm_XL_2().to(device)
    network.train()

    out = network(img, t, y=y, y_image=y_image, use_image_num=use_image_num)
    print(out.shape)