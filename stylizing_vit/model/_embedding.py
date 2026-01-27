"""
xAILab Bamberg
University of Bamberg

@description:
Adapted Position Embedding.

@references:
mae: https://github.com/facebookresearch/mae
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

# Import packages
import numpy as np


class PositionEmbedding:
    """
    Position embedding utils.
    """

    @staticmethod
    def get_2d_sincos_pos_embed(
        embed_dim: int, grid_size: int, cls_token=False
    ) -> np.array:
        """
        2D sine-cosine position embedding.

        Args:
            embed_dim (int): Embedding dimension.
            grid_size (int): Grid height and width.

        Returns:
            (np.array): Position embedding [grid_size*grid_size, embed_dim] or
            [1+grid_size*grid_size, embed_dim] (with or without cls_token)
        """

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = PositionEmbedding.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

        return pos_embed

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid) -> np.array:
        """
        Get the 2d sin-cosine position embedding from the provided grid.

        Args:
            embed_dim (int): Embedding dimension.
            grid (np.meshgrid): Grid height and width.

        Returns:
            (np.array): 2d sin-cosine position embedding
        """

        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = PositionEmbedding.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[0]
        )  # (H*W, D/2)
        emb_w = PositionEmbedding.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[1]
        )  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

        return emb

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.array) -> np.array:
        """
        Get the 1d sin-cosine position embedding from the provided grid.

        Args:
            embed_dim (int): output dimension for each position
            pos (np.array): a list of positions to be encoded: size (M,)

        Returns:
            (np.array): 1d sin-cosine position embedding.
        """

        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb
