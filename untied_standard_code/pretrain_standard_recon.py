"""

pretrain_standard_recon.py

     SPDX-License-Identifier: GNU GENERAL PUBLIC LICENSE
                              Version 3, 29 June 2007

     Copyright © 2026 Görkem Can Süleymanoğlu

     Standart Untied Dot-Product Attention

         This software is made available for academic research use only.
         Commercial use is prohibited without explicit permission from the
         copyright holder. This software includes pretrain_standard_recon.py
         and standard_mahalanobis.py as the core.

         GNU GENERAL PUBLIC LICENSE

         The GNU General Public License is a free, copyleft license for
         software and other kinds of works.

         The licenses for most software and other practical works are designed
         to take away your freedom to share and change the works.  By contrast,
         the GNU General Public License is intended to guarantee your freedom to
         share and change all versions of a program--to make sure it remains free
         software for all its users.  We, the Free Software Foundation, use the
         GNU General Public License for most of our software; it applies also to
         any other work released this way by its authors.  You can apply it to
         your programs, too.

         When we speak of free software, we are referring to freedom, not
         price.  Our General Public Licenses are designed to make sure that you
         have the freedom to distribute copies of free software (and charge for
         them if you wish), that you receive source code or can get it if you
         want it, that you can change the software or use pieces of it in new
         free programs, and that you know you can do these things.

         To protect your rights, we need to prevent others from denying you
         these rights or asking you to surrender the rights.  Therefore, you have
         certain responsibilities if you distribute copies of the software, or if
         you modify it: responsibilities to respect the freedom of others

         continues with...

->>  Uses ImageNet Subset (all INTEGER classes)
->>  Model: Untied Standard (encoder + decoder)
            ->>  precise architecture

->>  Output: full model weights (pretrained_standard_full_recon.pt)

"""
import os
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import seaborn as sns
from google.colab import drive
from dotenv import load_dotenv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import login
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

drive.mount("/content/drive", force_remount=True)

COLORS = {
    'recon': '#1f77b4', 'cls': '#d62728', 'proto': '#9467bd',
    'train': '#1f77b4', 'val': '#ff7f0e', 'lr': '#2ca02c',
    'best': '#e41a1c',
}


def _apply_style():
    sns.set_style("whitegrid", {"grid.linestyle": "--", "axes.edgecolor": "0.2"})

    sns.set_context("paper")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.2,
    })


def _dense_ticks(ax, x_major=None, y_major=None, x_minor=2, y_minor=2):
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))

    if y_major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))

    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))

    ax.grid(True, which='major', alpha=0.30, linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.12, linewidth=0.5)

    ax.tick_params(which='both', direction='out')


class StandardAttention(nn.Module):
    """
        Standard multi-head self-attention
        with scaled dot-product compatibility
        function.

        Ablation baseline against the tied Gloeba
        (energy-based attention) module. Instead
        of a per-head learnable bilinear matrix
        M and a learned temperature, affinity
        is computed as Q·Kᵀ / sqrt(d_h) with
        separate, untied projection matrices
        W_q and W_k.

        This is the canonical Transformer
        attention (Vaswani et al., 2017).
        With W_q ≠ W_k the effective per-head
        form W_{q,h} W_{k,h}^T is generally
        neither symmetric nor PSD, so this
        module already reaches the asymmetric
        /indefinite territory that Gloeba's M_h
        is designed for — see
        `test_untied_absorption` below, which
        measures exactly this. The symmetric-PSD
        restriction (Sym_+, Proposition 5) applies
        only to the TIED case (W_q = W_k); it is
        not a property of this untied baseline.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, zone):
        """
            Project tokens into
            Q, K, V via separate
            untied linear maps,
            compute scaled dot-product
            attention scores Q·Kᵀ /
            sqrt(d_h), apply softmax
            and dropout, aggregate V,
            then project the output
            back to the model
            dimension.

            Args:
                zone: input tokens
                of shape
                        (B, N, D).
            Returns:
                Tensor of shape
                        (B, N, D).
        """
        B, N, D = zone.shape

        H, Dh = self.num_heads, self.head_dim

        Q = self.W_q(zone).reshape(B, N, H, Dh).permute(0, 2, 1, 3)
        K = self.W_k(zone).reshape(B, N, H, Dh).permute(0, 2, 1, 3)
        V = self.W_v(zone).reshape(B, N, H, Dh).permute(0, 2, 1, 3)

        attn = F.softmax(Q @ K.transpose(-2, -1) * (Dh ** -0.5), dim=-1)

        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, D)

        return self.proj(out)


class PrototypeCrossAttention(nn.Module):
    """
        Cross-attention that summarises a
        variable number of patch tokens into
        a fixed set of learnable prototype
        vectors.

        The prototypes act as the queries
        while the patch features provide
        the keys and values, so the output
        always has `num_prototypes` slots
        regardless of how many patches are
        fed in. Standard scaled dot-product
        attention is used here.
    """

    def __init__(self, dim, num_prototypes=64, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.num_prototypes = num_prototypes

        self.prototypes = nn.Parameter(torch.randn(1, num_prototypes, dim))

        nn.init.trunc_normal_(self.prototypes, std=0.02)

        self.q_proj = nn.Linear(dim, dim)

        self.kv_proj = nn.Linear(dim, dim * 2)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_feats):
        """
            Use the learnable prototypes as queries
            and the patch features as keys/values,
            run scaled dot-product cross-attention,
            and return one aggregated vector per
            prototype.

            Args:
                patch_feats: patch tokens
                of shape (B, N, D).
            Returns:
                Tensor of shape (B,
                num_prototypes, D).
        """
        B, N, D = patch_feats.shape
        H, Dh = self.num_heads, self.head_dim

        P = self.num_prototypes

        proto = self.prototypes.expand(B, -1, -1)

        Q = self.q_proj(proto).reshape(B, P, H, Dh).permute(0, 2, 1, 3)
        kv = self.kv_proj(patch_feats).reshape(B, N, 2, H, Dh).permute(2, 0, 3, 1, 4)

        K, V = kv[0], kv[1]

        attn = (Q @ K.transpose(-2, -1)) * (Dh ** -0.5)

        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, P, D)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
        Pre-norm Transformer block:
        LayerNorm → StandardAttention →
        residual,

        then LayerNorm → GELU MLP →
        residual.
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.attn = StandardAttention(dim, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, lemma):
        """
            Apply the pre-norm attention
            sublayer and the pre-norm
            feed-forward sublayer, each
            wrapped in a residual
            connection.

            Args:
                lemma: input tokens
                of shape (B, N, D).
            Returns:
                Tensor of shape
                (B, N, D).
        """
        lemma = lemma + self.attn(self.norm1(lemma))
        lemma = lemma + self.ffn(self.norm2(lemma))

        return lemma


class MultiScalePatchEmbed(nn.Module):
    """
        Patch embedding that operates
        at several patch sizes at once.

        For each patch size the image
        is split into non-overlapping
        patches, flattened, linearly
        projected to `embed_dim`, and
        given its own learnable
        positional embeddings.

        The token sequences from
        all scales are concatenated,
        so the model sees both
        fine-grained (small patches)
        and coarse (large patches)
        structure.
    """

    def __init__(self, img_size=64, in_chans=3, embed_dim=512, patch_sizes=(8,), dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.embeds = nn.ModuleList()

        self.token_counts = []

        for ps in patch_sizes:
            n_tok = (img_size // ps) ** 2
            patch_dim = ps * ps * in_chans

            self.embeds.append(nn.Sequential(
                nn.Linear(patch_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            ))

            self.token_counts.append(n_tok)

        self.total_tokens = sum(self.token_counts)

        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n, embed_dim)) for n in self.token_counts
        ])

        for pe in self.pos_embeds:
            nn.init.trunc_normal_(pe, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

    def forward(self, cousin):
        """
            Split the image into patches at
            every set up scale, embed each
            scale and add its positional
            embedding, then concatenate
            all token sequences.

            Args:
                cousin: input images
                of shape (B, C, H, W).
            Returns:
                Token tensor of shape (B,
                total_tokens, embed_dim).
        """
        B = cousin.shape[0]

        image_input = cousin.permute(0, 2, 3, 1).contiguous()

        # (B, C, H, W) -> (B, H, W, C)

        tokens_list = []

        for ps, embed, pe in zip(self.patch_sizes, self.embeds, self.pos_embeds):
            patches = image_input.unfold(1, ps, ps).unfold(2, ps, ps)

            patches = patches.contiguous().view(B, -1, ps * ps * image_input.shape[-1])

            tok = embed(patches) + pe
            tokens_list.append(tok)

        return torch.cat(tokens_list, dim=1)


class TransformerDecoder(nn.Module):
    """
        Transformer-based decoder that
        mirrors the encoder architecture.

        Args:
            latent_dim (int): Dimension of the input
                              latent vector (default: 384)
            embed_dim (int): Internal token dimension for
                             transformer (default: 512)
            num_patches (int): Number of output patches/tokens
                               (default: 256)
            num_heads (int): Number of attention heads per block
                             (default: 8)
            num_layers (int): Number of transformer blocks
                              (default: 6)
            target_size (int): Output image size (height/width)
                               in pixels (default: 64)
            num_channels (int): Number of image channels
                                (default: 3)
            dropout_rate (float): Dropout probability
                                  (default: 0.1)
    """

    def __init__(
            self,
            latent_dim: int = 384,
            embed_dim: int = 512,
            num_patches: int = 256,
            num_heads: int = 8,
            num_layers: int = 6,
            target_size: int = 64,
            num_channels: int = 3,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.target_size = target_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        patches_per_side = int(math.sqrt(self.num_patches))
        self.actual_patch_size = self.target_size // patches_per_side

        self.latent_projector = nn.Linear(latent_dim, embed_dim)

        self.decoder_queries = nn.Parameter(
            torch.randn(1, num_patches, embed_dim)
        )

        nn.init.trunc_normal_(self.decoder_queries, std=0.02)

        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches, embed_dim)
        )

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.patch_to_pixels = nn.Sequential(
            nn.Linear(embed_dim,
                      int(int(self.actual_patch_size) * int(self.actual_patch_size) * int(self.num_channels)))
        )

        self.token_specific_latent = nn.Linear(latent_dim, num_patches * embed_dim)

        self.cross_attn_norm_q = nn.LayerNorm(embed_dim)

        # Query normalization

        self.cross_attn_norm_kv = nn.LayerNorm(embed_dim)

        # Key/Value normalization

        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

        self.adaptive_weight = nn.Parameter(torch.ones(1))

    def forward(self, latent_vector: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        """
            Reconstruct an image
            from a latent vector.

            Args:
                latent_vector: Input latent
                code of shape (B, latent_dim)
                encoder_features: Optional
                encoder features for
                cross-attention
                (B, N, embed_dim)

            Returns:
                Reconstructed image of shape
                (B, num_channels, target_size,
                target_size)
        """
        batch_size = latent_vector.shape[0]

        latent_condition = self.latent_projector(latent_vector).unsqueeze(1)

        decoder_tokens = self.decoder_queries.expand(batch_size, -1, -1)
        decoder_tokens = decoder_tokens + self.positional_embedding

        # Token-specific latent

        token_specific = self.token_specific_latent(latent_vector).reshape(batch_size, self.num_patches, self.embed_dim)

        decoder_tokens = decoder_tokens + token_specific * self.adaptive_weight

        decoder_tokens = decoder_tokens + latent_condition

        if encoder_features is not None:
            # Pre-norm: normalize
            # before attention

            q = self.cross_attn_norm_q(decoder_tokens)
            k = self.cross_attn_norm_kv(encoder_features)
            v = self.cross_attn_norm_kv(encoder_features)

            attn_out, _ = self.cross_attention(q, k, v)
            decoder_tokens = decoder_tokens + attn_out

            # Residual connection

        for block_type in self.transformer_blocks:
            decoder_tokens = block_type(decoder_tokens)

        decoder_tokens = self.final_layer_norm(decoder_tokens)

        patch_pixels = self.patch_to_pixels(decoder_tokens)

        patch_grid = patch_pixels.reshape(
            batch_size, self.num_patches, self.actual_patch_size, self.actual_patch_size, self.num_channels
        )

        patches_per_side = int(math.sqrt(self.num_patches))
        assert patches_per_side * patches_per_side == self.num_patches

        # 8 for 64 patches

        image_grid = patch_grid.reshape(
            batch_size,
            patches_per_side,
            patches_per_side,
            self.actual_patch_size,
            self.actual_patch_size,
            self.num_channels
        )

        reconstructed = image_grid.permute(0, 1, 3, 2, 4, 5).contiguous()

        reconstructed = reconstructed.view(
            batch_size,
            patches_per_side * self.actual_patch_size,
            patches_per_side * self.actual_patch_size,
            self.num_channels
        )

        reconstructed = reconstructed.permute(0, 3, 1, 2)

        # (B, 3, 64, 64)

        reconstructed = F.interpolate(
            reconstructed,
            size=(self.target_size, self.target_size),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )

        return reconstructed


class SELYNE(nn.Module):
    """
        Full Untied Standard encoder–decoder backbone. Sign: pretrain_standard_recon.py

        Pipeline: (for pretrain_standard_recon.py)
        multiscale patch embedding → prepend a CLS token and add
        global positional embeddings → a stack of standard
        Transformer blocks → pooling via the CLS token
        concatenated with prototype-cross-attention features
        → projection to a compact latent `z`. From `z`, a decoder
        reconstructs the input image and a linear classifier produces
        class logits. The classifier weights double as the semantic
        class prototypes used by the prototype-matching loss.
    """

    def __init__(
            self,
            img_size=64,
            in_chans=3,
            embed_dim=256,
            latent_dim=128,
            num_heads=4,
            depth=3,
            num_prototypes=64,
            patch_sizes=(8,),
            mlp_ratio=4.0,
            dropout=0.1,
            num_classes=200):
        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans

        self.patch_embed = MultiScalePatchEmbed(img_size, in_chans, embed_dim, patch_sizes, dropout)

        total_tokens = self.patch_embed.total_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.global_pos = nn.Parameter(torch.zeros(1, total_tokens + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.global_pos, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.proto_attn = PrototypeCrossAttention(embed_dim, num_prototypes, num_heads, dropout)

        proto_out_dim = num_prototypes * embed_dim
        combined_dim = embed_dim + proto_out_dim

        self.encoder_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.decoder = TransformerDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_patches=total_tokens,
            num_heads=num_heads,
            num_layers=depth,
            target_size=img_size,
            num_channels=in_chans,
            dropout_rate=dropout
        )

        self.classifier = nn.Linear(latent_dim, num_classes)

        self.proto_log_tau = nn.Parameter(torch.tensor(float(np.log(0.1))))

    @property
    def semantic_prototypes(self):
        """
            Expose the classifier weight
            matrix (num_classes × latent_dim)
            as the set of semantic class prototypes,
            so the prototype-matching loss and the
            classifier share the same parameters.
        """
        return self.classifier.weight

    def encode(self, world):
        """
            Encode an image batch into
            the latent representation `z`.

            Runs patch embedding, the
            Transformer stack and the
            prototype cross-attention,
            then fuses the CLS token
            with the flattened prototype
            features through the encoder
            head.

            Args:
                world: input images
                of shape (B, C, H, W).
            Returns:
                Latent tensor `z` of
                shape (B, latent_dim).
        """
        if world.dim() == 2:
            world = world.view(world.shape[0], self.in_chans, self.img_size, self.img_size)

        B = world.shape[0]
        tokens = self.patch_embed(world)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.pos_drop(tokens + self.global_pos)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]
        patch_out = tokens[:, 1:]

        proto_flat = self.proto_attn(patch_out).reshape(B, -1)

        z_demand = self.encoder_head(torch.cat([cls_out, proto_flat], dim=-1))

        return z_demand, patch_out

    def decode(self, z_val, encoder_features):
        """
            Map a latent vector back to
            an image reconstruction via
            the convolutional decoder.
            Returns a tensor of shape
            (B, C, H, W).
        """
        return self.decoder(z_val, encoder_features)

    def classify(self, z_val):
        """
            Produce class logits from
            a latent vector using the
            linear classifier head.
        """
        return self.classifier(z_val)

    def forward(self, x_val):
        """
            Full forward pass: encode
            the image into `z`, decode
            it into a reconstruction,
            and reshape the output to
            (B, 3, img_size, img_size).

            Returns:
                (x_reconstruction, z)
                — the reconstructed
                image and its latent
                code.
        """
        zeta, encoder_feats = self.encode(x_val)

        x_reconstruction_flat = self.decode(zeta, encoder_feats)

        x_reconstruction = x_reconstruction_flat.view(-1, 3, self.img_size, self.img_size)

        return x_reconstruction, zeta

    def score_components(self, sea_1, Y_reconstruction):
        """
            Compute score components:
              l1 — mean pixel-wise MSE,
              l2 — absolute difference of total-variation
                   (edge/smoothness mismatch),
              l3 — MSE between log-magnitude 2D FFT spectra
                   (frequency-domain mismatch),
              l4 — mean of locally pooled squared error
                   (coarse spatial mismatch).

            Args:
                sea_1: input images, shape (B, C * H * W)
                or reshape-compatible.
                Y_reconstruction: reconstructions of the
                same shape.
            Returns:
                Tensor of shape (B, 4) stacking the
                four components per sample.
        """
        B = sea_1.shape[0]

        x_img = sea_1.reshape(B, self.in_chans, self.img_size, self.img_size)
        r_img = Y_reconstruction.reshape(B, self.in_chans, self.img_size, self.img_size)

        l1 = F.mse_loss(r_img, x_img, reduction='none').reshape(B, -1).mean(dim=1)

        def tv_per_sample(img_1):
            dh = (img_1[:, :, 1:, :] - img_1[:, :, :-1, :]).abs().reshape(B, -1).mean(dim=1)
            dw = (img_1[:, :, :, 1:] - img_1[:, :, :, :-1]).abs().reshape(B, -1).mean(dim=1)
            return dh + dw

        l2 = (tv_per_sample(x_img) - tv_per_sample(r_img)).abs()

        fft_x = torch.fft.fft2(x_img).abs()
        fft_r = torch.fft.fft2(r_img).abs()

        l3 = F.mse_loss(torch.log1p(fft_r), torch.log1p(fft_x), reduction='none').reshape(B, -1).mean(dim=1)

        assert self.img_size % 8 == 0, f"Image size {self.img_size} not divisible by 8"
        l4 = F.avg_pool2d((x_img - r_img) ** 2, kernel_size=8, stride=8).reshape(B, -1).mean(dim=1)

        return torch.stack([l1, l2, l3, l4], dim=1)

    def compute_loss(self, zeta, Z_reconstruction):
        """
            Training objective: take the four
            reconstruction components, apply
            the (currently uniform) per-component
            weights, sum them per sample and average
            over the batch into a single scalar loss.

            Args:
                zeta: input images.
                Z_reconstruction: their
                                  reconstructions.
            Returns:
                Scalar reconstruction loss.
        """
        comp = self.score_components(zeta, Z_reconstruction)

        weights = torch.ones(4, device=comp.device).unsqueeze(0)

        return (weights * comp).sum(dim=1).mean()


class FlatImageDataset(Dataset):
    """
        Return one sample. The (random)
        transform is applied here at access
        time, so a fresh augmentation is sampled
        every epoch. Returns just the image when
        no labels were provided, otherwise an
        (image, label) pair.
    """

    def __init__(self, extremities, phrases=None, transform_left=None):
        self.images = extremities
        self.labels = phrases

        self.transform = transform_left

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_land = self.images[idx]

        if self.transform is not None:
            image_land = self.transform(image_land)

            # -> (C, H, W) tensor

        if self.labels is None:
            return image_land

        return image_land, self.labels[idx]


def load_all_subset_imagenet(split="train"):
    """
        Returns resized RGB PIL images
        (NOT tensors) so the Dataset can
        apply fresh random augmentation
        each epoch.
    """
    print(f"Loading Subset ImageNet ({split} split)...")

    ds = load_dataset("zh-plus/tiny-imagenet", split=split)

    images_input, labels_input = [], []

    for sample in ds:
        img_input = sample["image"]

        if not isinstance(img_input, Image.Image):
            img_input = Image.fromarray(img_input)

        img_input = img_input.convert("RGB").resize((64, 64))

        images_input.append(img_input)

        labels_input.append(sample["label"])

    print(f"Loaded {len(images_input)} images as input.")

    return images_input, labels_input


def test_untied_absorption(modeled, threshold=0.1):
    """
        Ablation diagnostic:
        checks whether the
        effective interaction
        matrix W_q^T @ W_k
        per head is symmetric
        or asymmetric.

        Symmetric → untied
        attention collapses
        to tied-symmetric
        territory (no gain
        over tied Gloeba with
        M ∈ Sym_+). Asymmetric
        → untied attention
        captures directional
        token interactions.
    """
    for name, module in modeled.named_modules():

        if isinstance(module, StandardAttention):
            W_q = module.W_q.weight.detach().cpu()
            W_k = module.W_k.weight.detach().cpu()

            H = module.num_heads
            Dh = module.head_dim

            print(f"\n{name}:")

            for h in range(H):
                Wq_h = W_q[h * Dh:(h + 1) * Dh, :]
                Wk_h = W_k[h * Dh:(h + 1) * Dh, :]

                M_eff = Wq_h @ Wk_h.T

                # (Dh x Dh) effective
                # interaction matrix

                sym = 0.5 * (M_eff + M_eff.T)
                asym = 0.5 * (M_eff - M_eff.T)

                asym_norm = torch.norm(asym, p='fro').item()
                sym_norm = torch.norm(sym, p='fro').item()
                asym_ratio = asym_norm / (sym_norm + 1e-10)

                eigenvalues = torch.linalg.eigvalsh(sym)
                is_psd = torch.all(eigenvalues >= -1e-6).item()

                sv = torch.linalg.svdvals(M_eff)
                condition = sv[0].item() / (sv[-1].item() + 1e-10)

                if asym_ratio < threshold and is_psd:
                    territory = "symmetric psd  → tied-attention equivalent"
                elif asym_ratio < threshold and not is_psd:
                    territory = "symmetric indefinite (Krein space)"
                elif asym_ratio >= threshold and is_psd:
                    territory = "asymmetric psd"
                else:
                    territory = "asymmetric + indefinite (full untied territory)"

                print(f"  Head {h:3d}: {territory}")
                print(f"    Asymmetry ratio : {asym_ratio:.4f}  (threshold={threshold})")
                print(f"    Sym  Frob norm  : {sym_norm:.4f}")
                print(f"    Asym Frob norm  : {asym_norm:.4f}")
                print(f"    Min eigenvalue  : {eigenvalues.min():.4f}")
                print(f"    Condition number: {condition:.2f}")
                print(f"    In Sym_+        : {is_psd}")


def plot_professional_graphs(
        epochs_list_x,
        train_losses_list_y,
        recon_losses_list_z,
        cls_losses_list_t,
        proto_losses_list_i,
        train_accs_list_j,
        val_accs_list_k,
        lrs_list_a,
        best_acc_val_b,
        best_epoch_val_c,
        W_RECON_d,
        W_CLS_e,
        W_PROTO_f):
    """
        Generate four publication-grade
        training graphs the SELYNE model.
        Styling is consistent with the
        all other files (Mahalanobis
        experiments): seaborn whitegrid
        + dense ticks + despine.
    """
    _apply_style()

    PLOTS_DIR = "/content/drive/MyDrive/pre_training_plots_second"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    x_origin_pipe = np.asarray(epochs_list_x, dtype=float)

    train_l = np.asarray(train_losses_list_y, dtype=float)
    recon_l = np.asarray(recon_losses_list_z, dtype=float)

    cls_l = np.asarray(cls_losses_list_t, dtype=float)

    proto_l = np.asarray(proto_losses_list_i, dtype=float)
    train_a = np.asarray(train_accs_list_j, dtype=float)
    val_a_x = np.asarray(val_accs_list_k, dtype=float)

    lrs = np.asarray(lrs_list_a, dtype=float)

    n = len(x_origin_pipe)

    single = n < 2

    mk = 'o' if n <= 10 else None
    mk2 = 's' if mk else None

    best_epoch = int(best_epoch_val_c)
    has_best = 0 <= best_epoch < n

    # Figure 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        (axes[0, 0], train_l, 'Total Training Loss', COLORS['train']),
        (axes[0, 1], recon_l, 'Reconstruction Loss (MSE)', '#2ca02c'),
        (axes[1, 0], cls_l, 'Classification Loss', COLORS['cls']),
        (axes[1, 1], proto_l, 'Prototype Matching Loss', COLORS['proto']),
    ]

    for ax, ydata, title, color in panels:
        ax.plot(x_origin_pipe, ydata, color=color, linewidth=2, marker=mk, markersize=6)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        _dense_ticks(ax, x_minor=2, y_minor=2)

        sns.despine(ax=ax)

    fig.tight_layout()

    fig.savefig(os.path.join(PLOTS_DIR, 'fig1_loss_dashboard.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, 'fig1_loss_dashboard.pdf'), bbox_inches='tight')

    plt.close(fig)

    print("[\u2713] Figure 1 saved: fig1_loss_dashboard.png/pdf")

    # Figure 2

    assert len(lrs) == len(train_a) == len(val_a_x) == len(x_origin_pipe) \
        , "Epochs and metrics length mismatch"

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_origin_pipe, train_a, color=COLORS['train'], label='Train Acc',
             linewidth=2, marker=mk, markersize=6, zorder=3)
    ax1.plot(x_origin_pipe, val_a_x, color=COLORS['cls'], label='Val Acc',
             linewidth=2, marker=mk, markersize=6, zorder=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(x_origin_pipe, lrs, color=COLORS['lr'], linestyle='--', label='LR',
             linewidth=1.5, alpha=0.8, marker=mk2, markersize=5, zorder=2)
    ax2.set_ylabel('Learning Rate', color=COLORS['lr'])
    ax2.tick_params(axis='y', labelcolor=COLORS['lr'])
    ax2.grid(False)

    # don't clutter
    # the secondary
    # axis with a
    # grid

    if np.all(lrs > 0) and (lrs.max() / max(lrs.min(), 1e-12)) > 1.0:
        ax2.set_yscale('log')

    _dense_ticks(ax1, y_major=0.1, x_minor=2, y_minor=2)

    # Best point: clear star
    # marker + boxed label
    # parked in an empty
    # corner. The curved
    # (arc3) arrow guarantees
    # the annotation NEVER
    # crosses the curves.

    if has_best:
        ax1.scatter(best_epoch, best_acc_val_b, color=COLORS['best'], s=110,
                    marker='*', edgecolors='white', linewidths=1.2, zorder=6)
        ax1.annotate(
            f'Best Val Acc\n{best_acc_val_b:.3f} @ ep {best_epoch}',
            xy=(best_epoch, best_acc_val_b),
            xytext=(0.97, 0.18), textcoords='axes fraction',
            ha='right', va='center', fontsize=9.5, color=COLORS['best'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=COLORS['best'], alpha=0.95),
            arrowprops=dict(arrowstyle='->', color=COLORS['best'],
                            connectionstyle='arc3,rad=0.25', linewidth=1.4),
            zorder=7,
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(h1 + h2, l1 + l2, loc='center right', frameon=True, fontsize=10)
    ax1.set_title('SELYNE: Accuracy & Learning Rate', fontweight='bold')

    sns.despine(ax=ax1, top=True, right=False)

    # keep the right
    # spine (LR axis)

    fig.tight_layout()

    fig.savefig(os.path.join(PLOTS_DIR, 'fig2_accuracy_lr.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, 'fig2_accuracy_lr.pdf'), bbox_inches='tight')

    plt.close(fig)

    print("[\u2713] Figure 2 saved: fig2_accuracy_lr.png/pdf")

    # Figure 3

    fig, ax = plt.subplots(figsize=(10, 6.5))

    w_recon = W_RECON_d * recon_l
    w_cls = W_CLS_e * cls_l
    w_proto = W_PROTO_f * proto_l
    total = w_recon + w_cls + w_proto

    total = np.where(total < 1e-8, 1.0, total)

    recon_norm = w_recon / total
    cls_norm = w_cls / total
    proto_norm_1 = w_proto / total

    ax.stackplot(x_origin_pipe, recon_norm, cls_norm, proto_norm_1,
                 labels=['Reconstruction', 'Classification', 'Prototype'],
                 colors=[COLORS['recon'], COLORS['cls'], COLORS['proto']], alpha=0.7)

    total_weight = W_RECON_d + W_CLS_e + W_PROTO_f

    targets = [
        (W_RECON_d / total_weight, COLORS['recon'], 'Recon target', (0, (6, 2))),
        (W_CLS_e / total_weight, COLORS['cls'], 'Cls target', (0, (3, 3))),
        (W_PROTO_f / total_weight, COLORS['proto'], 'Proto target', (0, (1, 2))),
    ]

    for y_val, color, lbl, dash in targets:
        ax.axhline(y=y_val, color=color, linestyle=dash, linewidth=1.8, label=lbl)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted Loss Contribution Ratio')
    ax.set_title('Loss Component Contributions (weighted)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.margins(x=0)

    _dense_ticks(ax, y_major=0.2, x_minor=2, y_minor=2)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)

    sns.despine(ax=ax)

    fig.savefig(os.path.join(PLOTS_DIR, 'fig3_loss_contribution.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, 'fig3_loss_contribution.pdf'), bbox_inches='tight')

    plt.close(fig)

    print("[\u2713] Figure 3 saved: fig3_loss_contribution.png/pdf")

    # Figure 4

    fig, (axc, axg) = plt.subplots(1, 2, figsize=(14, 5))

    axc.plot(x_origin_pipe, val_a_x, color=COLORS['val'], linewidth=2, marker=mk, markersize=6)
    axc.fill_between(x_origin_pipe, 0, val_a_x, alpha=0.12, color=COLORS['val'])

    if has_best:
        axc.scatter(best_epoch, best_acc_val_b, color=COLORS['best'], s=220,
                    marker='*', edgecolors='white', linewidths=1.2, zorder=5)

    axc.set_xlabel('Epoch')
    axc.set_ylabel('Validation Accuracy')
    axc.set_title('(a) Model Convergence', fontweight='bold')
    axc.set_ylim(bottom=0)

    _dense_ticks(axc, x_minor=2, y_minor=2)

    sns.despine(ax=axc)

    if not single:
        loss_grad = np.gradient(train_l)

        axg.plot(x_origin_pipe, loss_grad, color=COLORS['recon'], linewidth=1.5, marker=mk, markersize=5)
        axg.axhline(y=0, color='black', linewidth=0.8)
        axg.fill_between(x_origin_pipe, 0, loss_grad, where=(loss_grad > 0),
                         interpolate=True, alpha=0.3, color='red', label='Increasing')
        axg.fill_between(x_origin_pipe, 0, loss_grad, where=(loss_grad <= 0),
                         interpolate=True, alpha=0.3, color='green', label='Decreasing')
        axg.set_title('(b) Loss Change Rate', fontweight='bold')
        axg.legend(loc='best')
    else:
        axg.text(0.5, 0.5, 'Insufficient data\nfor gradient calculation\n(min 2 epochs needed)',
                 transform=axg.transAxes, ha='center', va='center', fontsize=12)
        axg.set_title('(b) Loss Change Rate (Insufficient Data)', fontweight='bold')

    axg.set_xlabel('Epoch')
    axg.set_ylabel('Loss Gradient')

    _dense_ticks(axg, x_minor=2, y_minor=2)

    sns.despine(ax=axg)

    fig.tight_layout()

    fig.savefig(os.path.join(PLOTS_DIR, 'fig4_convergence.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, 'fig4_convergence.pdf'), bbox_inches='tight')

    plt.close(fig)

    print("[\u2713] Figure 4 saved: fig4_convergence.png/pdf\n")


def set_seed(param_value=0):
    random.seed(param_value)
    np.random.seed(param_value)
    torch.manual_seed(param_value)
    torch.cuda.manual_seed_all(param_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(1337)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH = 64
    EPOCHS = 80
    LR = 5e-4
    WARMUP = 5
    PATIENCE = 20

    # loss weights and
    # regularization knobs

    MIXITUP_P = 0.5
    MIXITUP_ALPHA = 0.2
    LABEL_SMOOTH = 3e-2

    print(f"Using device: {DEVICE}")

    model = SELYNE(
        img_size=64,
        in_chans=3,
        embed_dim=512,
        latent_dim=384,
        num_heads=8,
        depth=6,
        num_prototypes=32,
        patch_sizes=(8,)
    ).to(DEVICE)

    # Transforms: augmentation on
    # train, deterministic on val

    NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        NORM,
        transforms.RandomErasing(p=0.25),
    ])

    val_tf = transforms.Compose([transforms.ToTensor(), NORM])
    train_images, train_labels = load_all_subset_imagenet(split="train")
    train_dataset = FlatImageDataset(train_images, train_labels, transform_left=train_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    val_images, val_labels = load_all_subset_imagenet(split="valid")

    val_loader = DataLoader(
        FlatImageDataset(val_images, val_labels, transform_left=val_tf),
        batch_size=BATCH, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)


    # cosine schedule
    # with linear warmup,
    # stepped once per epoch

    def lr_lambda(e):
        if e < WARMUP:
            return (e + 1) / WARMUP

        prog = (e - WARMUP) / max(1, EPOCHS - WARMUP)

        return 0.5 * (1.0 + np.cos(np.pi * prog))


    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    test_untied_absorption(model)

    best_acc = 0.0
    bad_epochs = 0

    # Metric lists for
    # the visualization

    epochs_list = []

    train_losses_list = []
    recon_losses_list = []
    cls_losses_list = []

    proto_losses_list = []
    train_accs_list = []
    val_accs_list = []
    lrs_list = []

    print(f"Pretraining started | Parameters: {sum(p.numel() for p in model.parameters()):,} | Epochs: {EPOCHS}")

    for ep in range(EPOCHS):
        model.train()

        total_loss, total_recon, total_cls, total_proto = 0.0, 0.0, 0.0, 0.0
        total_correct, total_seen = 0, 0

        lam = 1.0
        y_b = None
        x_mix = None

        # tracked on non-mix
        # up batches only

        loop = tqdm(train_loader, desc=f"Epoch {ep}", leave=True)

        for x, y in loop:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            # Mix up augmentation

            use_mix = np.random.rand() < MIXITUP_P

            if use_mix:
                lam = float(np.random.beta(MIXITUP_ALPHA, MIXITUP_ALPHA))
                perm = torch.randperm(x.size(0), device=DEVICE)
                y_b = y[perm]

                x_mix = lam * x + (1.0 - lam) * x[perm]

                x_recon, z = model(x_mix)
            else:
                x_recon, z = model(x)

            logits = model.classify(z)

            # Reconstruction loss

            recon_loss = model.compute_loss(x_mix if use_mix else x, x_recon)

            # Classification loss

            if use_mix:
                cls_loss = (lam * F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)
                            + (1.0 - lam) * F.cross_entropy(logits, y_b, label_smoothing=LABEL_SMOOTH))
            else:
                cls_loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)

            # Prototype loss

            z_norm = F.normalize(z, dim=1)

            proto_norm = F.normalize(model.semantic_prototypes, dim=1)
            tau = model.proto_log_tau.exp().clamp(min=0.01, max=1.0)

            proto_sim = (z_norm @ proto_norm.T) / tau

            if use_mix:
                proto_loss = (lam * F.cross_entropy(proto_sim, y)
                              + (1.0 - lam) * F.cross_entropy(proto_sim, y_b))
            else:
                proto_loss = F.cross_entropy(proto_sim, y)

            # ADDITIVE loss
            # — never multiply
            # losses together

            loss = recon_loss + cls_loss + proto_loss

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            opt.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_cls += cls_loss.item()
            total_proto += proto_loss.item()

            if not use_mix:
                total_correct += (logits.argmax(1) == y).sum().item()
                total_seen += y.size(0)

            loop.set_postfix(
                loss=loss.item(),
                acc=(total_correct / total_seen) if total_seen else 0.0,
            )

        sched.step()

        # Validation

        model.eval()

        with torch.no_grad():
            correct = 0

            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                _, z = model(x)

                logits = model.classify(z)
                correct += (logits.argmax(1) == y).sum().item()

            val_acc = correct / len(val_loader.dataset)
        train_acc = total_correct / max(1, total_seen)

        # Store metrics
        # for visualization

        epochs_list.append(ep)

        train_losses_list.append(total_loss / len(train_loader))
        recon_losses_list.append(total_recon / len(train_loader))
        cls_losses_list.append(total_cls / len(train_loader))
        proto_losses_list.append(total_proto / len(train_loader))

        train_accs_list.append(train_acc)
        val_accs_list.append(val_acc)

        lrs_list.append(opt.param_groups[0]['lr'])

        print(f"Ep {ep:3d} | Loss {total_loss / len(train_loader):.4f} | "
              f"Recon {total_recon / len(train_loader):.4f} | "
              f"Cls {total_cls / len(train_loader):.4f} | "
              f"Proto {total_proto / len(train_loader):.4f} | "
              f"Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f} | "
              f"LR {opt.param_groups[0]['lr']:.2e}")

        # Model checkpointing
        # and early stopping

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0

            torch.save(model.state_dict(), "/content/drive/MyDrive/pretrained_standard_full_recon_best.pt")

            print(f"  -> Saved best model (val_acc={val_acc:.3f})")
        else:
            bad_epochs += 1

            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {ep} (best val_acc={best_acc:.3f})")

                break

    print("\nAfter Training, Untied Standard Territory Analysis:")
    test_untied_absorption(model)

    # Save final model

    torch.save(model.state_dict(), "/content/drive/MyDrive/pretrained_standard_full_recon.pt")

    print(f"Pretraining finished → pretrained_standard_full_recon.pt (best val_acc: {best_acc:.3f})")

    # Generate professional
    # visualization plots

    print("\n")
    print("Generating professional visualization plots...")

    best_epoch_val = np.argmax(val_accs_list)
    best_acc_val = val_accs_list[best_epoch_val]

    plot_professional_graphs(
        epochs_list, train_losses_list, recon_losses_list,
        cls_losses_list, proto_losses_list, train_accs_list,
        val_accs_list, lrs_list, best_acc_val, best_epoch_val,
        1.0, 1.0, 1.0
    )

    print("\n   The figures were saved successfully into the pre_training_plots (second) folder!")
    print("     - fig1_loss_dashboard.png/pdf (4-panel loss visualization)")
    print("     - fig2_accuracy_lr.png/pdf (Accuracy + Learning Rate schedule)")
    print("     - fig3_loss_contribution.png/pdf (Loss component contribution ratios)")
    print("     - fig4_convergence.png/pdf (Convergence analysis with gradient)")

# Untied Standard Codex - Core