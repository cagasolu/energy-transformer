"""

standard_mahalanobis.py

     SPDX-License-Identifier: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
     Copyright © 2026 Görkem Can Süleymanoğlu

     Standard Untied Dot-Product Attention

         1.) Mahalanobis Distance requires pretrained weights from:

             ->>  pretrain_standard_recon.py

"""
import gc
import os
import math
import copy
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import seaborn as sns
from dotenv import load_dotenv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from huggingface_hub import login
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve
import matplotlib.patches as mpatches
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from scipy.ndimage import gaussian_filter1d
import torchvision.transforms as transforms
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def set_seed(param_value=0):
    random.seed(param_value)
    np.random.seed(param_value)

    torch.manual_seed(param_value)

    torch.cuda.manual_seed_all(param_value)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


COLORS = {
    'score': '#1f77b4', 'mahal': '#d62728',
    'normal': '#2ca02c', 'anomaly': '#d62728',
    'mean': '#111111', 'diag': '#888888',
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
        Full Untied Standard encoder–decoder backbone. Sign: standard_mahalanobis.py

        Pipeline: (for standard_mahalanobis.py)
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

    def score_diaphone(self, T, T_reconstruction):
        """
            Inference-time anomaly score: 
            the weighted sum of the four 
            reconstruction components per 
            sample, computed under `no_grad`.
            Higher score indicates a poorer
            reconstruction and thus a more 
            likely anomaly.

            Args:
                T: input images.
                T_reconstruction: their reconstructions.
            Returns:
                Per-sample standard tensor of shape (B,).
        """
        with torch.no_grad():
            comp = self.score_components(T, T_reconstruction)

            weights = torch.ones(4, device=comp.device).unsqueeze(0)

            return (weights * comp).sum(dim=1)


class FlatImageDataset(Dataset):
    """
        Return one sample. The 
        (random) transform is 
        applied here at access 
        time, so a fresh 
        augmentation is 
        sampled every epoch. 
        Returns just the image 
        when no labels were 
        provided, otherwise 
        an (image, label) 
        pair.
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


def load_dataset_for_anomaly(class_id=0, diff_samples=10000):
    """
        Load STL-10 for anomaly detection 
        with one-class classification 
        setup.

        Args:
            class_id (int): The class 
            label_set to treat as normal 
            (0-9 for STL-10)

            diff_samples (int): Maximum 
            number of anomaly samples
            to return. Default is 10000 
            (set to None to use all 
            7200)

        Returns:
            tuple: (train_norm, 
            test_norm, test_anom)
    """

    print(f"Loading STL-10 with normal class: {class_id}")

    # Transform pipeline: STL-10 native 96x96
    # -> resize to 64x64, convert to tensor,
    # normalize with ImageNet stats

    transform_dist = transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.LANCZOS, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Download and load
    # STL-10 datasets

    train_dataset_dist = STL10(root="./data", split="train", download=True, transform=transform_dist)
    test_dataset = STL10(root="./data", split="test", download=True, transform=transform_dist)

    train_norm = [img_value.reshape(-1) for img_value, label_set in train_dataset_dist if label_set == class_id]

    test_norm = [img_value_2.reshape(-1) for img_value_2, label_set2 in test_dataset if label_set2 == class_id]

    # 800 samples

    test_anom = [img_value_2.reshape(-1) for img_value_2, label_set3 in test_dataset if label_set3 != class_id]

    # 7200 samples; Cap 
    # anomaly samples 
    # if specified

    if diff_samples is not None and len(test_anom) > diff_samples:
        test_anom = test_anom[:diff_samples]

    # Print dataset statistics

    print(f"Train normal ({class_id}): {len(train_norm)}")
    print(f"Test normal ({class_id}): {len(test_norm)}")
    print(f"Test anomaly: {len(test_anom)}")

    return train_norm, test_norm, test_anom


@torch.no_grad()
def compute_mahalanobis_stats(model_second, loader, device):
    """
        Estimate the latent-space 
        Gaussian statistics of the 
        normal class.

        Encodes every sample in the loader, 
        computes the empirical mean and a 
        Ledoit-Wolf shrinkage covariance, 
        adds diagonal regularization if 
        the covariance is severely 
        ill-conditioned (condition 
        number > 1e6), and 
        inverts it.

        Args:
            model_second: 
            SELYNE model 
            used as the 
            feature encoder.
            loader: 
            DataLoader of
            normal samples.
            device: torch 
            device.
        Returns:
            (mean, cov_inv) as torch tensors on `device`.
    """
    model_second.eval()
    features = []

    for x_lot in loader:
        x_lot = x_lot.to(device)
        zero, _ = model_second.encode(x_lot)
        features.append(zero.cpu().numpy())

    features = np.concatenate(features, axis=0)
    mean = np.mean(features, axis=0)

    lw = LedoitWolf().fit(features)
    cov = lw.covariance_

    eigenvalues = np.linalg.eigvalsh(cov)

    condition_number = eigenvalues.max() / max(eigenvalues.min(), 1e-10)

    if condition_number > 1e6:
        lambda_reg = eigenvalues.max() / 1e6 - eigenvalues.min()
        cov = cov + lambda_reg * np.eye(features.shape[1])

    cov_inv = np.linalg.inv(cov)

    return torch.tensor(mean, device=device, dtype=torch.float32), torch.tensor(cov_inv, device=device,
                                                                                dtype=torch.float32)


def mahalanobis_distance(zero_3, mean, cov_inv):
    """
        Compute the squared Mahalanobis 
        distance of each latent vector 
        from the class mean under the 
        given inverse covariance: 
        (z − μ)ᵀ Σ⁻¹ (z − μ).

        Args:
            zero_3: latent vectors 
            of shape (B, latent_dim).
            mean: class mean of shape 
            (latent_dim,).
            cov_inv: inverse 
            covariance of shape 
            (latent_dim, latent_dim).
        Returns:
            Per-sample distance 
            tensor of shape (B,).
    """
    delta = zero_3 - mean.unsqueeze(0)

    return torch.einsum('bi,ij,bj->b', delta, cov_inv, delta)


def _mean_roc(roc_list, n_grid=200, smooth=1.5):
    """
        Aggregate multiple per-class 
        ROC curves onto a common FPR
        grid: linearly interpolate 
        each TPR curve, average them, 
        lightly Gaussian-smooth the 
        mean, and clamp the endpoints 
        to (0,0) and (1,1).

        Args:
            roc_list: list of (fpr, 
            tpr) arrays, one per class.
            n_grid: number of FPR grid 
            points.
            smooth: Gaussian smoothing 
            sigma for the mean curve.
        Returns:
            (grid, mean_tpr, std_tpr) 
            arrays over the shared 
            FPR grid.
    """
    grid = np.linspace(0.0, 1.0, n_grid)

    tpr_s = []

    for fpr, tpr in roc_list:
        interp = np.interp(grid, fpr, tpr)
        interp[0] = 0.0
        tpr_s.append(interp)

    tpr_s = np.array(tpr_s)

    mean_tpr = gaussian_filter1d(tpr_s.mean(axis=0), sigma=smooth)

    mean_tpr = np.clip(mean_tpr, 0, 1)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0

    return grid, mean_tpr, tpr_s.std(axis=0)


def _norm_within_class(conclusions, method, label_domain, pct=(1, 99)):
    """
        Min-max normalise scores 
        within each class before 
        pooling across classes, 
        so that per-class score 
        ranges become comparable 
        on a single [0, 1] axis.

        Args:
            conclusions: list of 
            per-class result dicts.
            method: 'score' or 
            'mahalanobis'.
            label_domain: 
            'normal' or 
            'anomaly'.
        Returns:
            1D array of pooled, 
            within-class-normalised
            scores.
    """
    pooled = []

    for r in conclusions:
        n, a = r[method]['normal'], r[method]['anomaly']

        lo, hi = np.percentile(n, pct)

        rng = (hi - lo) if (hi - lo) > 1e-12 else 1.0

        if label_domain == 'normal':
            vals = (n - lo) / rng

            vals = np.clip(vals, 0.0, 1.0)
        else:
            vals = (a - lo) / rng

        pooled.append(vals)

    return np.concatenate(pooled)


def plot_anomaly_eval_graphs(conclusions, save_prefix="fig_eval", out_dir="figures_second"):
    """
        Produce the three evaluation 
        figures from the per-class 
        conclusions and save each
        as PNG + PDF:
            1. grouped AUROC bar chart 
            comparing Reconstruction 
            score vs Mahalanobis 
            (with means),
            2. side-by-side mean ROC 
            curves (per-class faint
            lines + ±1 std band),
            3. within-class-normalised 
            score distributions for 
            normal vs anomaly.

        Args:
            conclusions: list of 
            per-class dicts holding
            'score' and 'mahalanobis' 
            metrics.
            save_prefix: filename 
            prefix for the saved 
            figures.
            out_dir: directory the 
            figures are written to 
            (created if missing).
    """
    _apply_style()

    os.makedirs(out_dir, exist_ok=True)
    save_prefix = os.path.join(out_dir, save_prefix)

    classes = [r['class_id'] for r in conclusions]

    e_auroc = np.array([r['score']['auroc'] for r in conclusions])
    m_auroc = np.array([r['mahalanobis']['auroc'] for r in conclusions])

    # Figure — AUROC grouped bar

    fig, ax = plt.subplots(figsize=(max(8, math.ceil(1.2 * len(classes))), 6))
    ax.axhspan(0.9, 1.0, color='green', alpha=0.06, zorder=0)

    x_coord = np.arange(len(classes))

    width = 0.38

    b1 = ax.bar(x_coord - width / 2, e_auroc, width, color=COLORS['score'],
                edgecolor='black', linewidth=0.8, zorder=3)
    b2 = ax.bar(x_coord + width / 2, m_auroc, width, color=COLORS['mahal'],
                edgecolor='black', linewidth=0.8, hatch='//', zorder=3)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()

            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.axhline(e_auroc.mean(), color=COLORS['score'], linestyle='--', linewidth=1.4, alpha=0.85)
    ax.axhline(m_auroc.mean(), color=COLORS['mahal'], linestyle=':', linewidth=1.8, alpha=0.85)
    ax.axhline(0.5, color='gray', linestyle='-', linewidth=0.9, alpha=0.6)

    handles = [
        mpatches.Patch(facecolor=COLORS['score'], edgecolor='black', label='Score'),
        mpatches.Patch(facecolor=COLORS['mahal'], edgecolor='black', hatch='//', label='Mahalanobis'),
        plt.Line2D([0], [0], color=COLORS['score'], ls='--', label=f'Score mean ({e_auroc.mean():.3f})'),
        plt.Line2D([0], [0], color=COLORS['mahal'], ls=':', label=f'Mahal. mean ({m_auroc.mean():.3f})'),
        mpatches.Patch(facecolor='green', alpha=0.15, label='Strong (>=0.9)'),
    ]

    ax.legend(handles=handles, loc='lower right', frameon=True, fontsize=9)

    ax.set_xlabel('Class ID')
    ax.set_ylabel('AUROC')
    ax.set_title('Anomaly Detection AUROC: Score vs Mahalanobis', fontweight='bold')
    ax.set_xticks(x_coord)
    ax.set_xticklabels(classes)
    ax.set_ylim([0, 1.05])

    _dense_ticks(ax, y_major=0.1, y_minor=2)
    sns.despine(ax=ax)

    fig.tight_layout()

    fig.savefig(f'{save_prefix}1_auroc_bar.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{save_prefix}1_auroc_bar.pdf', bbox_inches='tight')

    plt.close(fig)

    print(f"[OK] saved {save_prefix}1_auroc_bar.png/pdf")

    # Figure — ROC curves

    fig, (axe, axm) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, method, color, title in [
        (axe, 'score', COLORS['score'], '(a) Reconstruction Score ROC'),
        (axm, 'mahalanobis', COLORS['mahal'], '(b) Mahalanobis ROC'),
    ]:
        roc_list = [(r[method]['fpr'], r[method]['tpr']) for r in conclusions]
        grid = np.linspace(0, 1, 200)

        for fpr, tpr in roc_list:
            sm = gaussian_filter1d(np.interp(grid, fpr, tpr), sigma=1.2)
            ax.plot(grid, np.clip(sm, 0, 1), color=color, alpha=0.22, linewidth=1.0, zorder=2)

        gr, mean_tpr, std_tpr = _mean_roc(roc_list)

        mean_auc = np.mean([r[method]['auroc'] for r in conclusions])

        ax.plot(gr, mean_tpr, color=color, linewidth=2.8, zorder=4)
        ax.fill_between(gr, np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1), color=color, alpha=0.15, zorder=1)
        ax.plot([0, 1], [0, 1], color=COLORS['diag'], linestyle='--', linewidth=1.0, zorder=3)

        ax.text(0.97, 0.50, f'Mean AUROC = {mean_auc:.3f}\nClasses = {len(conclusions)}',
                transform=ax.transAxes, fontsize=10, va='center', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=color, alpha=0.9))

        handles = [
            plt.Line2D([0], [0], color=color, lw=2.8, label='Mean ROC'),
            mpatches.Patch(facecolor=color, alpha=0.15, label='+/-1 std'),
            plt.Line2D([0], [0], color=color, alpha=0.3, lw=1, label='Per-class'),
            plt.Line2D([0], [0], color=COLORS['diag'], ls='--', lw=1, label='Chance'),
        ]

        ax.legend(handles=handles, loc='lower right', fontsize=9, frameon=True)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        _dense_ticks(ax, x_major=0.2, y_major=0.2, x_minor=2, y_minor=2)
        sns.despine(ax=ax)

    fig.tight_layout()

    fig.savefig(f'{save_prefix}2_roc_curves.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{save_prefix}2_roc_curves.pdf', bbox_inches='tight')

    plt.close(fig)

    print(f"[OK] saved {save_prefix}2_roc_curves.png/pdf")

    # Figure — score distributions

    fig, (axe, axm) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, method, title in [
        (axe, 'score', '(a) Reconstruction Score Distribution'),
        (axm, 'mahalanobis', '(b) Mahalanobis Distance Distribution'),
    ]:
        norm_n = _norm_within_class(conclusions, method, 'normal')
        norm_a = _norm_within_class(conclusions, method, 'anomaly')

        bins = np.linspace(0, 1, 50)

        ax.hist(norm_n, bins=bins, color=COLORS['normal'], alpha=0.35, density=True,
                edgecolor='white', linewidth=0.3)
        ax.hist(norm_a, bins=bins, color=COLORS['anomaly'], alpha=0.35, density=True,
                edgecolor='white', linewidth=0.3)

        sns.kdeplot(norm_n, ax=ax, color=COLORS['normal'], linewidth=2.2, fill=False, clip=(0, 1))
        sns.kdeplot(norm_a, ax=ax, color=COLORS['anomaly'], linewidth=2.2, fill=False, clip=(0, 1))

        ax.axvline(norm_n.mean(), color=COLORS['normal'], linestyle='--', linewidth=1.6)
        ax.axvline(norm_a.mean(), color=COLORS['anomaly'], linestyle='--', linewidth=1.6)

        handles = [
            mpatches.Patch(facecolor=COLORS['normal'], alpha=0.6, label='Normal'),
            mpatches.Patch(facecolor=COLORS['anomaly'], alpha=0.6, label='Anomaly'),
        ]

        ax.legend(handles=handles, loc='upper center', fontsize=9, frameon=True)
        ax.set_xlabel('Normalized score (within-class robust p1–p99)')
        ax.set_ylabel('Density')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim([0, 1])

        _dense_ticks(ax, x_major=0.2, x_minor=2, y_minor=2)
        sns.despine(ax=ax)

    fig.tight_layout()

    fig.savefig(f'{save_prefix}3_score_dist.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{save_prefix}3_score_dist.pdf', bbox_inches='tight')

    plt.close(fig)

    print(f"[OK] saved {save_prefix}3_score_dist.png/pdf")

    print("\n   All evaluation figures saved successfully!")


@torch.no_grad()
def evaluate_mahalanobis(model_three, normal_loader, anomaly_loader, device, mean, cov_inv):
    """
        Args:
            model_three: 
            SELYNE encoder.
            normal_loader: 
            dataloader for 
            the normal class.
            anomaly_loader: 
            dataloader for 
            the anomalies.
            cov_inv: latent 
            inverse covariance.
            mean: latent class.
            device: torch device.
        Returns:
            Dict with auroc, 
            fpr, tpr, and the
            raw normal/anomaly
            distance arrays.
            :param model_three:
            :param normal_loader:
            :param anomaly_loader:
            :param device:
            :param mean:
            :param cov_inv:
    """
    model_three.eval()

    def collect_dists(loader):
        dists = []

        for x_lot_2 in loader:
            x_lot_2 = x_lot_2.to(device)
            zero_2, _ = model_three.encode(x_lot_2)

            d = mahalanobis_distance(zero_2, mean, cov_inv).cpu().numpy()

            dists.extend(d.tolist())

        return np.asarray(dists, dtype=float)

    d_normal = collect_dists(normal_loader)
    d_anomaly = collect_dists(anomaly_loader)

    y_true = np.array([0] * len(d_normal) + [1] * len(d_anomaly))
    scores = np.concatenate([d_normal, d_anomaly])

    auroc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    print(f"MAHALANOBIS -> AUROC: {auroc:.4f} | "
          f"Normal mean dist: {np.mean(d_normal):.4f} | Anomaly mean dist: {np.mean(d_anomaly):.4f}")

    return {'auroc': float(auroc),
            'fpr': fpr,
            'tpr': tpr,
            'normal': d_normal,
            'anomaly': d_anomaly}


@torch.no_grad()
def evaluate_score(model_forth, normal_loader, anomaly_loader, device):
    """
        Args:
            model_forth: 
            SELYNE model.
            normal_loader: 
            DataLoader for 
            the normal class.
            anomaly_loader: 
            DataLoader for 
            the anomaly class.
            device: torch device.
        Returns:
            Dict with auroc, 
            fpr, tpr, and the
            raw normal/anomaly
            score arrays.
            :param model_forth:
            :param normal_loader:
            :param anomaly_loader:
            :param device:
    """
    model_forth.eval()

    def collect_scores(loader):
        scores_tie = []

        for x_lot_3 in loader:
            x_lot_3 = x_lot_3.to(device)
            recon, _ = model_forth(x_lot_3)

            e_value = model_forth.score_diaphone(x_lot_3, recon)

            scores_tie.extend(e_value.cpu().numpy().tolist())

        return np.asarray(scores_tie, dtype=float)

    e_normal = collect_scores(normal_loader)
    e_anomaly = collect_scores(anomaly_loader)

    y_true = np.array([0] * len(e_normal) + [1] * len(e_anomaly))
    scores = np.concatenate([e_normal, e_anomaly])

    auroc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    print(f"Reconstruction Score -> AUROC: {auroc:.4f} | "
          f"Normal mean score: {np.mean(e_normal):.4f} | Anomaly mean score: {np.mean(e_anomaly):.4f}")

    return {'auroc': float(auroc),
            'fpr': fpr,
            'tpr': tpr,
            'normal': e_normal,
            'anomaly': e_anomaly}


def augment_batch(x, img_size, in_chans, pad=32):
    """
        Light, tensor-based
        augmentation (function)
        to produce two views.
    """
    B = x.shape[0]

    ImageMeetHall = x.view(B, in_chans, img_size, img_size)

    flip = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1, 1)
    ImageMeetHall = torch.where(flip, ImageMeetHall.flip(-1), ImageMeetHall)

    sh, sw = random.randint(-2, 2), random.randint(-2, 2)

    ImageMeetHall = F.pad(ImageMeetHall, (pad, pad, pad, pad), mode='reflect')

    ImageMeetHall = ImageMeetHall[:, :, pad + sh: pad + sh + img_size, pad + sw: pad + sw + img_size]

    return ImageMeetHall


@torch.no_grad()
def ema_update(student, teacher, decay):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(decay).add_(ps.data, alpha=1 - decay)

    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


def latent_consistency(z_s, z_t, var_gamma=0.5):
    """
        BYOL invariance 
        (cosine distance) 
        + VICReg
        variance hinge 
        (per-dimension 
        std dev penalty).
    """
    z_s_n = F.normalize(z_s, dim=-1)
    z_t_n = F.normalize(z_t, dim=-1)

    invariance = 2 - 2 * (z_s_n * z_t_n.detach()).sum(dim=-1).mean()

    # variance reg on the raw latent —
    # the same space Mahalanobis uses

    std = torch.sqrt(z_s.var(dim=0, unbiased=False) + 1e-4)

    variance = F.relu(var_gamma - std).mean()

    return invariance, variance


def train_mahalanobis_enhanced(model_five,
                               train_loader_class,
                               device,
                               epochs=80,
                               lr=1e-4,
                               wd=1e-4,
                               ema_decay=0.99):
    """
        Args:
            model_five: 
            pretrained 
            SELYNE model.
            train_loader_class: 
            DataLoader of the 
            single normal 
            class. epochs, 
            lr, wd, ema_decay: 
            optimisation 
            hyperparameters.
            device: torch 
            device.
        Returns:
            The fine-tuned model.
            :param model_five:
            :param train_loader_class:
            :param device:
            :param epochs:
            :param lr:
            :param wd:
            :param ema_decay:
    """
    for propose in model_five.parameters():
        propose.requires_grad = True

    K_o = math.log(5)

    params = list(model_five.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-7)

    # EMA teacher = slow-moving copy of
    # pretrained geometry (anchor +
    # momentum)

    teacher = copy.deepcopy(model_five).to(device).eval()

    for propose in teacher.parameters():
        propose.requires_grad = False

    img_size, in_chans = model_five.img_size, model_five.in_chans

    model_five.train()

    for epoch in range(epochs):
        tot_e = tot_c = tot_std = 0.0

        n = 0

        for xorn in train_loader_class:
            xorn = xorn.to(device)

            # Two views: ->

            v1 = augment_batch(xorn, img_size, in_chans)
            v2 = augment_batch(xorn, img_size, in_chans)

            # student: view1 -> recon
            # (score) + latent

            recon, z_s = model_five(v1)
            recon_loss = model_five.compute_loss(v1, recon)

            # teacher (frozen, momentum):
            # view2 -> latent target

            with torch.no_grad():
                z_t, _ = teacher.encode(v2)

            inv_loss, var_loss = latent_consistency(z_s, z_t)
            consist_loss = inv_loss + var_loss

            loss_input = K_o * recon_loss + (math.pi / 10) * consist_loss

            optimizer.zero_grad()
            loss_input.backward()

            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)

            optimizer.step()

            # momentum update: teacher's
            # knowledge slowly follows the
            # student

            ema_update(model_five, teacher, ema_decay)

            tot_e += recon_loss.item()
            tot_c += consist_loss.item()

            tot_std += z_s.detach().std(dim=0).mean().item()

            n += 1

        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch:3d} | Score: {tot_e / n:.4f} | Consist: {tot_c / n:.4f} | "
              f"LatentStd: {tot_std / n:.4f} | LR: {lr_now:.2e}")

    return model_five


@torch.no_grad()
def save_reconstructions(model_sixth, loader, device, out_dir="reconstructions",
                         max_samples=20, show_metrics=True):
    """
        Reconstruct images from a 
        loader and save original 
        | reconstruction pairs 
        as PNGs with quality 
        metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    model_sixth.eval()

    # same normalization stats used at load time

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    total = len(loader.dataset)
    n_pick = min(max_samples, total)

    chosen = set(random.Random().sample(range(total), n_pick))

    global_idx = 0

    saved = 0

    all_psnr = []

    # Create subdirectories for
    # different visualizations

    compare_dir = os.path.join(out_dir, "comparison")
    diff_dir = os.path.join(out_dir, "difference")

    os.makedirs(compare_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)

    for x_input_0 in loader:
        x_input_0 = x_input_0.to(device)

        if x_input_0.dim() == 2:
            x_input_0 = x_input_0.view(-1, model_sixth.in_chans,
                                       model_sixth.img_size, model_sixth.img_size)

        recon, zwe = model_sixth(x_input_0)

        recon = recon.reshape(-1, model_sixth.in_chans, model_sixth.img_size, model_sixth.img_size)

        # back to [0,1]

        x_img = (x_input_0 * std + mean).clamp(0, 1)
        r_img = (recon * std + mean).clamp(0, 1)

        # Calculate difference map

        diff_map = torch.abs(x_img - r_img)

        # Normalize difference
        # for visualization

        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

        for i in range(x_input_0.shape[0]):

            # Track global index,
            # skip anything not
            # randomly selected

            idx = global_idx
            global_idx += 1

            if idx not in chosen:
                continue

            # Calculate PSNR

            mse = F.mse_loss(r_img[i], x_img[i]).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(max(mse, 1e-10)))

            all_psnr.append(psnr)

            # Create comparison image
            # (original | reconstruction)

            pair = torch.cat([x_img[i], r_img[i]], dim=2)
            pair_arr = (pair.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Save comparison

            Image.fromarray(pair_arr).save(os.path.join(compare_dir, f"recon_{saved:04d}_psnr_{psnr:.1f}dB.png"))

            # Create difference heatmap

            diff_vis = (diff_map[i] * 255).byte().cpu().permute(1, 2, 0).numpy()

            # Convert single channel
            # to RGB for visualization

            if diff_vis.shape[-1] == 1:
                diff_vis = np.repeat(diff_vis, 3, axis=-1)

            Image.fromarray(diff_vis).save(os.path.join(diff_dir, f"diff_{saved:04d}_psnr_{psnr:.1f}dB.png"))

            # Also, save the individual and original
            # outputs along with the reconstruction

            orig_arr = (x_img[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            recon_arr = (r_img[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            Image.fromarray(orig_arr).save(os.path.join(out_dir, f"original_{saved:04d}.png"))
            Image.fromarray(recon_arr).save(os.path.join(out_dir, f"reconstruction_{saved:04d}.png"))

            print(f"  Saved sample {saved:04d} | PSNR: {psnr:.2f} dB | MSE: {mse:.6f}")

            saved += 1

        if saved >= n_pick:
            break

    # Summary

    if show_metrics and len(all_psnr) > 0:
        print(f"\n Final Reconstruction Quality Summary:")
        print(f"  Total samples: {len(all_psnr)}")
        print(f"  Average PSNR: {np.mean(all_psnr):.2f} "
              f"  ± {np.std(all_psnr):.2f} dB")
        print(f"  Min PSNR: {np.min(all_psnr):.2f} dB")
        print(f"  Max PSNR: {np.max(all_psnr):.2f} dB")
        print(f"  Median PSNR: {np.median(all_psnr):.2f} dB")


if __name__ == "__main__":
    BATCH = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_seed_value = 2584

    set_seed(param_seed_value)
    print(f"Device: {DEVICE}")

    # Load all STL-10 classes one by
    # one, train SELYNE, and evaluate

    CLASS_LIST = list(range(10))
    waitingTop_results = []

    for CLASS_ID in CLASS_LIST:
        print(f"Processing Class ID: {CLASS_ID}")

        train_n, test_n, test_a = load_dataset_for_anomaly(CLASS_ID)

        train_dataset = FlatImageDataset(train_n)

        test_normal_dataset = FlatImageDataset(test_n)
        test_anomaly_dataset = FlatImageDataset(test_a)

        train_loader = DataLoader(train_dataset, BATCH, shuffle=True, num_workers=2)

        test_normal_loader = DataLoader(test_normal_dataset, BATCH, shuffle=False, num_workers=2)
        test_anomaly_loader = DataLoader(test_anomaly_dataset, BATCH, shuffle=False, num_workers=2)

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

        checkpoint_path = "pretrained_standard_full_recon.pt"

        if not os.path.exists(checkpoint_path):
            print(f"Warning: {checkpoint_path} not found. Using random init.")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint, strict=True)
            print("Pretrained model loaded.")

        print("\nFine-tuning with Mahalanobis enhancement")

        model = train_mahalanobis_enhanced(model, train_loader, DEVICE, epochs=10, lr=6e-4)

        mean_lat, cov_inv_lat = compute_mahalanobis_stats(model, train_loader, DEVICE)

        print("\nReconstruction Score Detection")

        score_results = evaluate_score(model, test_normal_loader, test_anomaly_loader, DEVICE)

        print("\nMahalanobis Distance Detection")

        mahalanobis_results = evaluate_mahalanobis(model, test_normal_loader, test_anomaly_loader, DEVICE, mean_lat,
                                                   cov_inv_lat)

        waitingTop_results.append({
            'class_id': CLASS_ID,
            'score': score_results,
            'mahalanobis': mahalanobis_results
        })

        torch.save({
            'encoder_head': model.encoder_head.state_dict(),
            'proto_attn': model.proto_attn.state_dict(),
            'mean': mean_lat.cpu(),
            'cov_inv': cov_inv_lat.cpu(),
        }, f"selyne_class_{CLASS_ID}.pt")

        print(f"\n✓ Model saved for class {CLASS_ID}")

        save_reconstructions(model, test_normal_loader, DEVICE,
                             out_dir=f"recon_class_{CLASS_ID}_normal",
                             show_metrics=True)

        save_reconstructions(model, test_anomaly_loader, DEVICE,
                             out_dir=f"recon_class_{CLASS_ID}_anomaly",
                             show_metrics=True)

        del model
        del train_loader
        del test_normal_loader
        del test_anomaly_loader
        del mean_lat
        del cov_inv_lat

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    plot_anomaly_eval_graphs(waitingTop_results, save_prefix="stl10_eval")

    print("FINAL RESULTS SUMMARY")

    for result in waitingTop_results:
        print(f"Class {result['class_id']}: "
              f"Reconstruction Score AUROC={result['score']['auroc']:.4f}, "
              f"Mahalanobis AUROC={result['mahalanobis']['auroc']:.4f}")

# STL-10 experiments deal with 
# the Untied Standard Codex; but, 
# the all code scalable and 
# modifiable.