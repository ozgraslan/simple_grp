"""
Simple GRP
Based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
"""

import numpy as np
import torch
from torch import nn

## this function is taken from https://github.com/huggingface/lerobot/blob/145fe4cd17971ea27188df23b2fe9d87e8e5128b/lerobot/common/policies/act/modeling_act.py#L685
def create_sinusoidal_pos_embedding(num_positions, dimension):
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.hidden_dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.dim_head * config.num_heads
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.to_qkv = nn.Linear(config.hidden_dim, inner_dim * 3, bias = False)
        self.mha = nn.MultiheadAttention(
            inner_dim, config.num_heads, 
            dropout=0, bias=False, batch_first=True
        )
        self.to_out = nn.Linear(inner_dim, config.hidden_dim, bias = False)

    def forward(self, x, att_mask = None, return_att=False):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        ## From https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        ## For a binary mask, a True value indicates that the corresponding position is not allowed to attend. 
        out, att = self.mha(q, k, v, need_weights=return_att, attn_mask=att_mask, average_attn_weights=False)
        return self.to_out(out), att    

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.layers = nn.ModuleList([])
        for _ in range(config.depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(config),
                    FeedForward(config)
            ]))

    def forward(self, x, att_mask = None, return_att=False):
        attention_maps = []
        for attn, ff in self.layers:
            o, w  = attn(x, att_mask, return_att=return_att) 
            if return_att:
                attention_maps.append(w)
            x = o + x
            x = ff(x) + x
        return self.norm(x), tuple(attention_maps)

class ContinuousActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, x):
        return self.mlp(x)

class LinearProjectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.text_projection = nn.Linear(config.text_dim, config.hidden_dim)
        self.image_projection = nn.Linear(config.image_dim, config.hidden_dim)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)

    def forward(self, x):
        text_emb  = self.text_projection(x["text"])
        image_emb = self.image_projection(x["image"])
        state_emb = self.state_projection(x["state"]).unsqueeze(1)
        return torch.cat([text_emb, image_emb, state_emb], dim=1)


ACTIONHEADDICT = {"continuous": ContinuousActionHead}
PROJLAYERDICT = {"linear": LinearProjectionLayer}


class SimpleGRP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pos_embedding = create_sinusoidal_pos_embedding(
            num_positions = config.num_tokens,
            dimension = config.hidden_dim,
        ) 
        self.register_buffer("pos_embedding", pos_embedding)
        self.transformer = Transformer(config)

        self.learnable_action_tokens = nn.Parameter(torch.randn(1, config.num_action_chunks, config.hidden_dim))
        self.action_head = ACTIONHEADDICT[config.action_head](config)
        self.projection_layer = PROJLAYERDICT[config.projection_layer](config)

    def forward(self, batch, return_att=False):
        input_tokens = self.projection_layer(batch)
        action_tokens = self.learnable_action_tokens.expand(input_tokens.size(0), -1, -1)
        tokens = torch.cat([input_tokens, action_tokens], dim=1)
        tokens = tokens + self.pos_embedding 
        att_mask = batch["att_mask"] if "att_mask" in batch else None
        out, att_map = self.transformer(tokens, att_mask = att_mask, return_att=return_att)
        return self.action_head(out[:, -self.config.num_action_chunks:, :]), att_map

    @torch.no_grad()
    def get_action(self, batch):
        # batch size is expected to be 1
        # chunk size is self.num_action_tokens
        # returns the first predicted action in the chunk
        action_chunk, _ = self(batch)
        return action_chunk[0].float().cpu().numpy()