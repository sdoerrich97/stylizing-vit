"""
xAILab Bamberg
University of Bamberg

@description:
Adapted Cross Attention blocks from timm and stable diffusion.

@references:
Attention: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

Cross Attention:
- https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/crossvit.py
- https://github.com/kjsman/stable-diffusion-pytorch/blob/main/stable_diffusion_pytorch/attention.py
- https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/attention.py#L145
"""

# Import packages
import torch
import torch.nn as nn
from timm.layers import Mlp, DropPath


class CrossAttention(nn.Module):
    """
    Cross Attention mechanism.
    """

    def __init__(
        self,
        embed_dim_input: int,
        embed_dim_context: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Constructor for the cross attention mechanism.

        Args:
            embed_dim_input (int): dimension of the input embedding
            embed_dim_context (int): dimension of the context/conditional
            information embedding
            num_heads (int): number of attention heads
            qkv_bias (bool): whether to add bias for the Q,K, and V matrices or not
            attn_drop (float): attention dropout
            proj_drop (float): projection dropout
        """
        super(CrossAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim_input // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim_input, embed_dim_input, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim_context, embed_dim_input, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim_context, embed_dim_input, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim_input, embed_dim_input)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Apply Cross Attention as softmax(Q_X * K_Y^T / sqrt(d)) * V_Y

        Args:
            X (torch.Tensor): feature embedding of the input
            Y (torch.Tensor): feature embedding of the context/conditional information

        Returns:
            (torch.Tensor) result of the cross attention between input
            and context/conditional information
        """

        B, N, L = X.shape
        interim_shape = (B, N, self.num_heads, self.head_dim)

        q = self.q_proj(X).view(interim_shape).transpose(1, 2)
        k = self.k_proj(Y).view(interim_shape).transpose(1, 2)
        v = self.v_proj(Y).view(interim_shape).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        XY = (attn @ v).transpose(1, 2).reshape(B, -1, L)
        XY = self.out_proj(XY)
        XY = self.proj_drop(XY)

        return XY


class CrossAttentionBlock(nn.Module):
    """
    Cross Attention Block adapted from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/crossvit.py
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim_input: int,
        embed_dim_context: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_input=nn.Module,
        norm_context=nn.Module,
        norm_input_name: str = "LayerNorm",
        norm_context_name: str = "LayerNorm",
    ):
        """
        Constructor for the cross attention block.

        Args:
            embed_dim_input (int): dimension of the input embedding
            embed_dim_context (int): dimension of the context/conditional
            information embedding
            num_heads (int): number of attention heads
            qkv_bias (bool): whether to add bias for the Q,K, and V matrices or not
            attn_drop (float): attention dropout
            proj_drop (float): projection dropout
            proj_drop (float): projection dropout
            norm_input (nn.Module): normalization layer for the input
            norm_context (nn.Module): normalization layer for the context/conditional
            information
        """

        super().__init__()

        if norm_input_name == "NoNorm":
            self.norm_input = nn.Identity()

        elif norm_input_name == "LayerNorm":
            self.norm_input = norm_input(embed_dim_input)

        elif norm_input_name == "InstanceNorm":
            self.norm_input = norm_input(num_patches)

        else:
            raise ValueError(f"Normalization layer {norm_input_name} is not supported.")

        if norm_context_name == "NoNorm":
            self.norm_context = nn.Identity()

        elif norm_context_name == "LayerNorm":
            self.norm_context = norm_context(embed_dim_context)

        elif norm_context_name == "InstanceNorm":
            self.norm_context = norm_context(num_patches)

        else:
            raise ValueError(
                f"Normalization layer {norm_context_name} is not supported."
            )

        self.attn = CrossAttention(
            embed_dim_input=embed_dim_input,
            embed_dim_context=embed_dim_context,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.mlp = Mlp(
            in_features=embed_dim_input,
            hidden_features=int(embed_dim_input * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """
        Apply Cross Attention as:   CA(X, Y) = softmax(Q(X) * K(Y)^T / sqrt(d)) * V(Y)
        and residual connection as: X = X + CA(X, Y)

        Args:
            X (torch.Tensor): feature embedding of the input
            Y (torch.Tensor): feature embedding of the context/conditional information

        Returns:
            (torch.Tensor) result of cross attention between input and
            context/conditional information plus residual
            connection of input
        """

        if Y is not None:
            X = X + self.drop_path(self.attn(self.norm_input(X), self.norm_context(Y)))
            X = X + self.drop_path(self.mlp(self.norm_input(X)))

        return X
