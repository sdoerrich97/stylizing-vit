"""
xAILab Bamberg
University of Bamberg

@description:
Encoder.
"""

# Imports
import torch
import torch.nn as nn
import timm
from functools import partial
from typing import Callable
from timm.models.vision_transformer import PatchEmbed

# Import own scripts
from stylizing_vit.model._embedding import PositionEmbedding
from stylizing_vit.model._attention import CrossAttentionBlock


class ViT(nn.Module):
    """
    Cross-attention based Vision Transformer encoder.
    """

    def __init__(
        self,
        input_size: int = 224,
        in_channel: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
    ):
        """
        Constructor.

        Args:
            input_size (int): Input size. Defaults to 224.
            in_channel (int): Number of input channels. Defaults to 3.
            patch_size (int): Patch size. Defaults to 16.
            embed_dim (int): Embedding dimension. Defaults to 768.
            depth (int): Depth of the model. Defaults to 12.
            num_heads (int): Number of heads. Defaults to 12.
            mlp_ratio (float): MLP ratio. Defaults to 4.0.
            qkv_bias (bool): QKV bias. Defaults to True.
            norm_layer (Callable): Normalization layer.
                Defaults to partial(nn.LayerNorm, eps=1e-6).
        """
        super(ViT, self).__init__()

        # Store the object attributes
        self.input_size = input_size
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.norm_layer = norm_layer

        # Create the patch embedding
        self.patch_embed = PatchEmbed(
            self.input_size, self.patch_size, self.in_channel, self.embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # Create the CLS token and the position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # CLS Token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=True
        )  # Create the position embedding, could be frozen as well if wanted by
        # setting requires_grad=False

        # Initialize the normalization layer
        if self.norm_layer == "nn.LayerNorm":
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Create the cross-attention blocks
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    num_patches=self.num_patches + 1,
                    embed_dim_input=embed_dim,
                    embed_dim_context=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    norm_input=self.norm_layer,
                    norm_context=self.norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        # Create the normalization layer
        self.norm = self.norm_layer(embed_dim)

        # Initialize the encoder's weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the encoder.
        """

        # Initialize pos_embed by sin-cos embedding
        pos_embed = PositionEmbedding.get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize nn.Linear and nn.LayerNorm.

        :param m: Module to initialize
        """

        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_input_size(self) -> int:
        """
        Get the input size of the encoder.
        """
        return self.input_size

    def get_patch_size(self) -> int:
        """
        Get the patch size of the encoder.
        """
        return self.patch_size

    def get_in_channel(self) -> int:
        """
        Get the number of input channels of the encoder.
        """
        return self.in_channel

    def get_embed_dim(self) -> int:
        """
        Get the embedding dimension of the encoder.
        """
        return self.embed_dim

    def forward(self, X: torch.Tensor, X_style: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            X (torch.Tensor): a 4d image tensor of shape [B, C, H, W]
            X_style (torch.Tensor): another 4d image tensor of shape [B, C, H, W] which
                style should be transferred to X

        Returns:
            Tuple(torch.Tensor): the output embeddings of the encoder
        """

        assert X.ndim == 4, "Input X must be four dimensional [B, C, H, W]"

        # Embed patches
        X, X_style = self.patch_embed(X), self.patch_embed(X_style)

        # Append cls token
        cls_tokens = self.cls_token.expand(X.shape[0], -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)
        X_style = torch.cat((cls_tokens.clone(), X_style), dim=1)

        # Add the pos embed
        X = X + self.pos_embed
        X_style = X_style + self.pos_embed

        # Apply the cross-attention blocks to do the style transfer
        X_stylized_X = X.clone()
        X_stylized_Y = X_style.clone()
        for i, blk in enumerate(self.blocks):
            X = blk(X, X)
            X_style = blk(X_style, X_style)

            X_stylized = blk(X_stylized_X, X_stylized_Y)
            X_stylized_X = blk(X, X_stylized)
            X_stylized_Y = X_style.clone()

        # Apply the normalization and store the output embedding of the encoder
        X, X_style, X_stylized = self.norm(X), self.norm(X_style), self.norm(X_stylized)

        # Return the embeddings
        return X, X_style, X_stylized


def prepare_encoder(backbone: str = "base", **kwargs) -> nn.Module:
    """
    Prepare the encoder with the specified backbone.

    Args:
        backbone (str): Backbone model name ("tiny", "small", "base", etc.).
        Defaults to "base".

        kwargs: Additional keyword arguments.
            - input_size (int): Input size. Defaults to 224.
            - in_channel (int): Number of input channels. Defaults to 3.
            - encoder_patch_size (int): Patch size. Defaults to 16.
            - encoder_embed_dim (int): Embedding dimension. Defaults to 768.
            - encoder_depth (int): Depth of the model. Defaults to 12.
            - encoder_num_heads (int): Number of heads. Defaults to 12.
            - encoder_mlp_ratio (float): MLP ratio. Defaults to 4.0.
            - encoder_qkv_bias (bool): QKV bias. Defaults to True.
            - encoder_norm_layer (Union[nn.Module, Callable]): Normalization layer.
            Defaults to partial(nn.LayerNorm, eps=1e-6).

    Returns:
        VisionTransformer: Vision Transformer model.
    """
    # Check the input types
    assert isinstance(backbone, str), "Backbone must be a string"

    if (
        backbone == "tiny"
        or backbone == "vit_tiny"
        or backbone == "vit_tiny_patch16_224"
    ):
        # Tiny ViT: input_size = 224, in_channel = 3, patch_size = 16, embed_dim = 192,
        # depth = 12, num_heads = 3, mlp_ratio = 4.0, norm_layer = 'nn.LayerNorm'
        return ViT(
            input_size=kwargs.get("input_size", 224),
            in_channel=kwargs.get("in_channel", 3),
            patch_size=kwargs.get("encoder_patch_size", 16),
            embed_dim=kwargs.get("encoder_embed_dim", 192),
            depth=kwargs.get("encoder_depth", 12),
            num_heads=kwargs.get("encoder_num_heads", 3),
            mlp_ratio=kwargs.get("encoder_mlp_ratio", 4.0),
            qkv_bias=kwargs.get("encoder_qkv_bias", True),
            norm_layer=kwargs.get(
                "encoder_norm_layer", partial(nn.LayerNorm, eps=1e-6)
            ),
        )

    elif (
        backbone == "small"
        or backbone == "vit_small"
        or backbone == "vit_small_patch16_224"
    ):
        # Small ViT: input_size = 224, in_channel = 3, patch_size = 16, embed_dim = 384,
        # depth = 12, num_heads = 6, mlp_ratio = 4.0, norm_layer = 'nn.LayerNorm'
        return ViT(
            input_size=kwargs.get("input_size", 224),
            in_channel=kwargs.get("in_channel", 3),
            patch_size=kwargs.get("encoder_patch_size", 16),
            embed_dim=kwargs.get("encoder_embed_dim", 384),
            depth=kwargs.get("encoder_depth", 12),
            num_heads=kwargs.get("encoder_num_heads", 6),
            mlp_ratio=kwargs.get("encoder_mlp_ratio", 4.0),
            qkv_bias=kwargs.get("encoder_qkv_bias", True),
            norm_layer=kwargs.get(
                "encoder_norm_layer", partial(nn.LayerNorm, eps=1e-6)
            ),
        )

    if (
        backbone == "base"
        or backbone == "vit_base"
        or backbone == "vit_base_patch16_224"
    ):
        # Base ViT: input_size = 224, in_channel = 3, patch_size = 16, embed_dim = 768,
        # depth = 12, num_heads = 12, mlp_ratio = 4.0, norm_layer = 'nn.LayerNorm'
        return ViT(
            input_size=kwargs.get("input_size", 224),
            in_channel=kwargs.get("in_channel", 3),
            patch_size=kwargs.get("encoder_patch_size", 16),
            embed_dim=kwargs.get("encoder_embed_dim", 768),
            depth=kwargs.get("encoder_depth", 12),
            num_heads=kwargs.get("encoder_num_heads", 12),
            mlp_ratio=kwargs.get("encoder_mlp_ratio", 4.0),
            qkv_bias=kwargs.get("encoder_qkv_bias", True),
            norm_layer=kwargs.get(
                "encoder_norm_layer", partial(nn.LayerNorm, eps=1e-6)
            ),
        )

    elif (
        backbone == "large"
        or backbone == "vit_large"
        or backbone == "vit_large_patch16_224"
    ):
        # Large ViT: input_size = 224, in_channel = 3, patch_size = 16,
        # embed_dim = 1024, depth = 24, num_heads = 16, mlp_ratio = 4.0,
        # norm_layer = 'nn.LayerNorm'
        return ViT(
            input_size=kwargs.get("input_size", 224),
            in_channel=kwargs.get("in_channel", 3),
            patch_size=kwargs.get("encoder_patch_size", 16),
            embed_dim=kwargs.get("encoder_embed_dim", 1024),
            depth=kwargs.get("encoder_depth", 24),
            num_heads=kwargs.get("encoder_num_heads", 16),
            mlp_ratio=kwargs.get("encoder_mlp_ratio", 4.0),
            qkv_bias=kwargs.get("encoder_qkv_bias", True),
            norm_layer=kwargs.get(
                "encoder_norm_layer", partial(nn.LayerNorm, eps=1e-6)
            ),
        )

    elif (
        backbone == "huge"
        or backbone == "vit_huge"
        or backbone == "vit_huge_patch14_224"
    ):
        # Huge ViT: input_size = 224, in_channel = 3, patch_size = 14, embed_dim = 1280,
        # depth = 32, num_heads = 16, mlp_ratio = 4.0, norm_layer = 'nn.LayerNorm'
        return ViT(
            input_size=kwargs.get("input_size", 224),
            in_channel=kwargs.get("in_channel", 3),
            patch_size=kwargs.get("encoder_patch_size", 14),
            embed_dim=kwargs.get("encoder_embed_dim", 1280),
            depth=kwargs.get("encoder_depth", 32),
            num_heads=kwargs.get("encoder_num_heads", 16),
            mlp_ratio=kwargs.get("encoder_mlp_ratio", 4.0),
            qkv_bias=kwargs.get("encoder_qkv_bias", True),
            norm_layer=kwargs.get(
                "encoder_norm_layer", partial(nn.LayerNorm, eps=1e-6)
            ),
        )

    else:
        # Create a custom ViT encoder model
        return ViT(
            input_size=kwargs.get("input_size", 224),
            in_channel=kwargs.get("in_channel", 3),
            patch_size=kwargs.get("encoder_patch_size", 16),
            embed_dim=kwargs.get("encoder_embed_dim", 768),
            depth=kwargs.get("encoder_depth", 12),
            num_heads=kwargs.get("encoder_num_heads", 12),
            mlp_ratio=kwargs.get("encoder_mlp_ratio", 4.0),
            qkv_bias=kwargs.get("encoder_qkv_bias", True),
            norm_layer=kwargs.get(
                "encoder_norm_layer", partial(nn.LayerNorm, eps=1e-6)
            ),
        )


def prepare_vgg_encoder() -> nn.Module:
    """
    Prepare the VGG19 encoder.

    Returns:
        nn.Module: VGG19 model.
    """

    return timm.create_model("vgg19", pretrained=True, features_only=True)
