"""
xAILab Bamberg
University of Bamberg

@description:
Main class for the Stylizing ViT model.
"""

# Imports
import torch
import torch.nn as nn
from typing import Tuple
from timm.layers import Mlp


# Import own scripts
from stylizing_vit.model.encoder import prepare_encoder, prepare_vgg_encoder
from stylizing_vit.loss import (
    Loss,
    compute_identity_loss,
    compute_consistency_loss,
    compute_anatomical_loss,
    compute_style_loss,
)
from stylizing_vit.util import unpatchify_image


class StylizingViT(nn.Module):
    def __init__(self, backbone: str = "base", train: bool = False, **kwargs):
        """
        Initialize the StylizingViT model.

        Args:
            backbone (str): The backbone architecture of the model
            ("tiny", "small", "base", etc.). Defaults to "base".
            train (bool): Whether the model is in training mode. Defaults to False.

            **kwargs: Additional parameters for the model, such as:
                - input_size (int, optional): Input size. Defaults to 224.
                - in_channel (int, optional): Number of input channels. Defaults to 3.
                - encoder_patch_size (int, optional): Patch size. Defaults to 16.
                - encoder_embed_dim (int, optional): Embedding dimension.
                Defaults to 768.
                - encoder_depth (int, optional): Depth of the model. Defaults to 12.
                - encoder_num_heads (int, optional): Encoder number of heads.
                Defaults to 12.
                - encoder_mlp_ratio (float, optional): Encoder MLP ratio.
                Defaults to 4.0.
                - encoder_qkv_bias (bool, optional): Encoder QKV bias. Defaults to True.
                - encoder_norm_layer (Callable, optional): Encoder normalization layer.
                Defaults to partial(nn.LayerNorm, eps=1e-6).
                - bottleneck_mlp_ratio (float, optional): Encoder MLP ratio.
                Defaults to 4.0.
                - bottleneck_act_layer (nn.Module, optional): Encoder MLP activation
                layer. Defaults to nn.GELU.
                - bottleneck_bias (bool, optional): Encoder MLP bias. Defaults to True.
                - bottleneck_drop (float, optional): Encoder MLP dropout.
                Defaults to 0.0.
                - post_process_conv_kernel_size (int, optional): Kernel size of the
                post-processing convolution. Defaults to 5.
                - post_process_conv_padding (int, optional): Padding of the
                post-processing convolution. Defaults to 2.
                - post_process_conv_bias (bool, optional): Whether to use bias in the
                post-processing convolution. Defaults to False.
        """

        # Initialize the parent constructor
        super(StylizingViT, self).__init__()

        # Initialize the encoder
        self.encoder = prepare_encoder(backbone=backbone, **kwargs)

        # Store the object parameters
        self.training_mode = train
        self.in_channel = self.encoder.get_in_channel()
        self.patch_size = self.encoder.get_patch_size()
        self.embed_dim = self.encoder.get_embed_dim()

        # Initialize the vgg encoder when the model is in training mode
        if self.training_mode:
            self.vgg_encoder = prepare_vgg_encoder()
            self.vgg_encoder.requires_grad_(False)
            self.vgg_encoder.eval()

        # Define the bottleneck
        self.bottleneck = Mlp(
            in_features=self.embed_dim,
            hidden_features=int(
                self.embed_dim * kwargs.get("bottleneck_mlp_ratio", 4.0)
            ),
            act_layer=kwargs.get("bottleneck_act_layer", nn.GELU),
            bias=kwargs.get("bottleneck_bias", True),
            drop=kwargs.get("bottleneck_drop", 0.0),
        )

        # Define the post-convolution
        self.post_process_conv = nn.Conv2d(
            self.in_channel,
            self.in_channel,
            kernel_size=kwargs.get("post_process_conv_kernel_size", 5),
            padding=kwargs.get("post_process_conv_padding", 2),
            bias=kwargs.get("post_process_conv_bias", False),
        )

    def reconstruct_image(
        self, Z_a: torch.tensor, Z_s: torch.tensor, remove_cls: bool = True
    ) -> torch.tensor:
        """
        Infuse the given style feature embeddings with the given anatomical features
        via Matrix-Multiplication in a patch-wise manner.

        Args:
            Z_a (torch.tensor): Anatomical feature embeddings of the input image(s)
            Z_s (torch.tensor): Style feature embeddings of the input image(s)
            remove_cls (bool): Whether to remove the CLS token from the input tensors

        Returns:
            S (torch.tensor): Synthesized image(s) [B, C, H, W]
        """

        # Remove the CLS token
        if remove_cls:
            Z_a = Z_a[:, 1:]  # [B, N, L]
            Z_s = Z_s[:, 1:]  # [B, N, L]

        # Reshape the input tensors to get the correct shape
        # for the matrix multiplication
        Z_a = Z_a.view(
            Z_a.size(0), Z_a.size(1), self.in_channel, self.patch_size, -1
        )  # [B, N, C, P, L]
        Z_s = Z_s.view(
            Z_s.size(0), Z_s.size(1), self.in_channel, -1, self.patch_size
        )  # [B, N, C, L, P]

        # Combine anatomical and style features for each patch of each sample
        # in the batch using matrix multiplication
        X_infused_patched = torch.matmul(Z_a, Z_s)  # [B, N, C, P, P]

        # Reshape the tensor to get the output image in the correct shape
        X_infused = unpatchify_image(
            X_infused_patched, patch_size=self.patch_size, in_channel=self.in_channel
        )  # [B, C, H, W]

        # Apply the post-processing convolution
        X_infused_post = self.post_process_conv(X_infused)

        # Return the infused image
        return X_infused_post

    def compute_loss(
        self,
        x: torch.Tensor,
        x_style: torch.Tensor,
        x_stylized: torch.Tensor,
        x_recon: torch.Tensor,
        x_style_recon: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the the individual loss terms, i.e. the identity, consistency,
        anatomical, and style loss.

        Args:
            x (torch.Tensor): Original image tensor of shape.
            x_style (torch.Tensor): Style image tensor of shape.
            x_stylized (torch.Tensor): Stylized image tensor of shape.
            x_recon (torch.Tensor): Reconstructed image tensor of shape.
            x_style_recon (torch.Tensor): Style reconstructed image tensor of shape.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
            containing the identity loss, consistency loss, anatomical loss,
            and style loss.
        """

        # Check the inputs
        assert isinstance(x, torch.Tensor), "Input x must be a torch.Tensor."
        assert isinstance(
            x_style, torch.Tensor
        ), "Input x_style must be a torch.Tensor."
        assert isinstance(
            x_stylized, torch.Tensor
        ), "Input x_stylized must be a torch.Tensor."
        assert isinstance(
            x_recon, torch.Tensor
        ), "Input x_recon must be a torch.Tensor."
        assert isinstance(
            x_style_recon, torch.Tensor
        ), "Input x_style_recon must be a torch.Tensor."
        assert x.dim() == 4, "Input x must have 4 dimensions (B, C, H, W)."
        assert x_style.dim() == 4, "Input x_style must have 4 dimensions (B, C, H, W)."
        assert (
            x_stylized.dim() == 4
        ), "Input x_stylized must have 4 dimensions (B, C, H, W)."
        assert x_recon.dim() == 4, "Input x_recon must have 4 dimensions (B, C, H, W)."
        assert (
            x_style_recon.dim() == 4
        ), "Input x_style_recon must have 4 dimensions (B, C, H, W)."

        # Pass the original, style, stylized, reconstructed, and style reconstructed
        # images through the vgg encoder
        z = self.vgg_encoder(x)
        z_style = self.vgg_encoder(x_style)
        z_stylized = self.vgg_encoder(x_stylized)
        z_recon = self.vgg_encoder(x_recon)
        z_style_recon = self.vgg_encoder(x_style_recon)

        # Compute the identity loss between the original and reconstructed images
        identity_loss = compute_identity_loss(x, x_recon) + compute_identity_loss(
            x_style, x_style_recon
        )

        # Compute the anatomical and style loss
        for i in range(len(z)):
            if i == 0:
                anatomical_loss = compute_anatomical_loss(z[i], z_stylized[i])
                style_loss = compute_style_loss(z_style[i], z_stylized[i])

            else:
                anatomical_loss += compute_anatomical_loss(z[i], z_stylized[i])
                style_loss += compute_style_loss(z_style[i], z_stylized[i])

        # Compute the consistncy loss
        consistency_loss = compute_consistency_loss(
            z[0], z_recon[0]
        ) + compute_consistency_loss(z_style[0], z_style_recon[0])

        for i in range(1, len(z)):
            consistency_loss += compute_consistency_loss(
                z[i], z_recon[i]
            ) + compute_consistency_loss(z_style[i], z_style_recon[i])

        # Create a Loss object to store the individual losses
        loss = Loss(
            identity=identity_loss,
            consistency=consistency_loss,
            anatomical=anatomical_loss,
            style=style_loss,
        )

        # Return the loss
        return loss

    def forward(
        self, x: torch.tensor, x_style: torch.tensor = None
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Forward pass of the StylizingViT model.

        Args:
            x (torch.tensor): Input image tensor of shape (B, C, H, W).
            x_style (torch.tensor, optional): Style image tensor of shape (B, C, H, W).
            If None, the style image is generated by rolling the input tensor.

        Returns:
            Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]: The stylized image
            tensor of shape (B, C, H, W) if in evaluation mode, or a tuple containing
            the loss and the reconstructed image tensor of shape (B, C, H, W)
            if in training mode.
        """

        # Check the inputs
        assert isinstance(x, torch.Tensor), "Input x must be a torch.Tensor."
        assert isinstance(
            x_style, (torch.Tensor, type(None))
        ), "Style input x_style must be a torch.Tensor or None."
        assert x.dim() == 4, "Input x must be a 4D tensor (B, C, H, W)."
        if x_style is not None:
            assert (
                x_style.dim() == 4
            ), "Style input x_style must have 4 dimensions (B, C, H, W)."

        # Prepare the style images if not provided
        if x_style is None:
            x_style = x.clone().roll(shifts=-1, dims=0)

        # Pass the input through the encoder
        # x, x_style: (B, C, H, W) -> z, z_style, z_stylized: (B, N, L)
        z, z_style, z_stylized = self.encoder(x, x_style)

        # Pass the stylized embeddings through the MLP-bottleneck
        # z_stylized: (B, N, L) -> z_stylized: (B, N, L)
        z_stylized = self.bottleneck(z_stylized)

        # Prepare the reconstruction by halving the stylized latent representation
        # z_stylized: (B, N, L) -> z_stylized_a: (B, N, L/2), z_stylized_b: (B, N, L/2)
        z_stylized_a, z_stylized_b = z_stylized.chunk(2, dim=-1)

        # Reconstruct the stylized image
        # z_stylized_a:(B, N, L/2) @ z_stylized_b:(B, N, L/2) ->
        # -> x_stylized: (B, C, H/P, W/P) -> x_stylized: (B, C, H, W)
        x_stylized = self.reconstruct_image(z_stylized_a, z_stylized_b)

        # When in training mode, compute the losses
        if self.training_mode:
            # Pass the encoded embeddings through the MLP-bottleneck
            # z, z_style: (B, N, L) -> z, z_style: (B, N, L)
            z = self.bottleneck(z)
            z_style = self.bottleneck(z_style)

            # Prepare the reconstruction by halving the encoded latent representation
            # z, z_style: (B, N, L) -> z_a: (B, N, L/2), z_b: (B, N, L/2)
            z_a, z_b = z.chunk(2, dim=-1)
            z_style_a, z_style_b = z_style.chunk(2, dim=-1)

            # Reconstruct the orginal image and the style image
            # z_a:(B, N, L/2) @ z_b:(B, N, L/2) ->
            # -> z_reconstructed: (B, C, H/P, W/P) ->
            # -> x_reconstructed: (B, C, H, W)
            x_recon = self.reconstruct_image(z_a, z_b)
            x_style_recon = self.reconstruct_image(z_style_a, z_style_b)

            # compute the loss terms
            loss = self.compute_loss(x, x_style, x_stylized, x_recon, x_style_recon)

            # Return the losses and the reconstructed images
            return loss, x_recon

        # Return the stylized images
        return x_stylized
