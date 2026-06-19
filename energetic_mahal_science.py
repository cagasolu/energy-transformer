"""

energetic_mahal_science.py

     SPDX-License-Identifier: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
     Copyright © 2026 Görkem Can Süleymanoğlu

     GLOBAL eba: global Energy Based Attention
         GLOEBA: Global Energy Based Attention
         —————————————————————————————————————
         SELYNE:    Stable-Energy Lyapunov Net

         1.) Mahalanobis Distance requires pretrained weights from:
             ->>  pretrain_selyne_recon.py  (model initialization)

         2.) One-dimensional version of globaleba_mahalanobis.py:
             ->>  globaleba_mahalanobis.py  (energy-based calculations methodology)

         3.) Usage: python or python3 energetic_mahal_science.py --data_root ./data/brisc2025

"""
import gc
import os
import math
import glob
import copy
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
from dotenv import load_dotenv
import torch.nn.functional as F
from huggingface_hub import login
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def set_seed(param_value=0):
    random.seed(param_value)
    np.random.seed(param_value)

    torch.manual_seed(param_value)

    torch.cuda.manual_seed_all(param_value)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


class EnergyBasedAttention(nn.Module):
    """
        Multi-head self-attention with a learned *energy* (bilinear) compatibility
        function instead of a plain dot product.

        For each head, query-key affinity is computed as a bilinear form Qᵀ·M·K,
        where M is a learnable (orthogonally initialised) per-head matrix, and
        the scores are scaled by a learnable per-head temperature before the
        softmax.

        This lets the model learn a richer, head-specific notion of token
        similarity than standard scaled dot-product attention.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        head_dim_obtain = int(self.head_dim)

        self.M = nn.Parameter(torch.randn(num_heads, head_dim_obtain, head_dim_obtain))

        nn.init.orthogonal_(self.M)

        self.log_temp = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(self, zone):
        """
            Project the input into Q, K, V; compute per-head bilinear energy
            scores (Q·M·Kᵀ), divide by the learned temperature, softmax into
            attention weights, aggregate V, then apply the output projection.

            Args:
                zone: input tokens of shape (B, N, D).
            Returns:
                Tensor of shape (B, N, D).
        """
        B, N, D = zone.shape
        H, Dh = self.num_heads, self.head_dim

        qkv = self.qkv(zone).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]

        score = torch.einsum('bhqd,hde,bhke->bhqk', Q, self.M, K)
        temp = self.log_temp.exp().view(1, H, 1, 1).clamp(min=1e-4)

        attn = F.softmax(score / temp, dim=-1)

        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, D)

        return self.proj(out)


class PrototypeCrossAttention(nn.Module):
    """
        Cross-attention that summarises a variable number of patch tokens into a fixed
        set of learnable prototype vectors.

        The prototypes act as the queries while the patch features provide the keys and
        values, so the output always has `num_prototypes` slots regardless of how many
        patches are fed in. Standard scaled dot-product attention is used here.
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
            Use the learnable prototypes as queries and the patch features
            as keys/values, run scaled dot-product cross-attention, and
            return one aggregated vector per prototype.

            Args:
                patch_feats: patch tokens of shape (B, N, D).
            Returns:
                Tensor of shape (B, num_prototypes, D).
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
        Pre-norm Transformer block: LayerNorm → EnergyBasedAttention
        → residual, then LayerNorm → GELU MLP → residual. The attention
        sub-layer uses the energy-based attention defined above rather
        than vanilla dot-product attention.
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.attn = EnergyBasedAttention(dim, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, lemma):
        """
            Apply the pre-norm attention sub-layer and the
            pre-norm feed-forward sub-layer, each wrapped
            in a residual connection.

            Args:
                lemma: input tokens of shape (B, N, D).
            Returns:
                Tensor of shape (B, N, D).
        """
        lemma = lemma + self.attn(self.norm1(lemma))
        lemma = lemma + self.ffn(self.norm2(lemma))

        return lemma


class MultiScalePatchEmbed(nn.Module):
    """
        Patch embedding that operates at several patch sizes at once.

        For each patch size the image is split into non-overlapping
        patches, flattened, linearly projected to `embed_dim`, and
        given its own learnable positional embeddings.

        The token sequences from all scales are concatenated,
        so the model sees both fine-grained (small patches)
        and coarse (large patches) structure.
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
            Split the image into patches at every configured scale, embed each
            scale and add its positional embedding, then concatenate all token
            sequences.

            Args:
                cousin: input images of shape (B, C, H, W).
            Returns:
                Token tensor of shape (B, total_tokens, embed_dim).
        """
        B = cousin.shape[0]
        image_input = cousin.permute(0, 2, 3, 1).contiguous()

        # (B,C,H,W) -> (B,H,W,C)

        tokens_list = []

        for ps, embed, pe in zip(self.patch_sizes, self.embeds, self.pos_embeds):
            patches = image_input.unfold(1, ps, ps).unfold(2, ps, ps)

            patches = patches.contiguous().view(B, -1, ps * ps * image_input.shape[-1])

            tok = embed(patches) + pe
            tokens_list.append(tok)

        return torch.cat(tokens_list, dim=1)


class TransformerDecoder(nn.Module):
    """
        Transformer-based decoder that mirrors the encoder architecture.

        Args:
            latent_dim (int): Dimension of the input latent vector (default: 384)
            embed_dim (int): Internal token dimension for transformer (default: 512)
            num_patches (int): Number of output patches/tokens (default: 256)
            num_heads (int): Number of attention heads per block (default: 8)
            num_layers (int): Number of transformer blocks (default: 6)
            target_size (int): Output image size (height/width) in pixels (default: 64)
            num_channels (int): Number of image channels (default: 3)
            dropout_rate (float): Dropout probability (default: 0.1)
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
                mlp_ratio=4.0,
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

        self.adaptive_weight = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, latent_vector: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        """
            Reconstruct an image from a latent vector.

            Args:
                latent_vector: Input latent code of shape (B, latent_dim)
                encoder_features: Optional encoder features for cross-attention (B, N, embed_dim)

            Returns:
                Reconstructed image of shape (B, num_channels, target_size, target_size)
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
            # Pre-norm: normalize before attention

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

        patches_per_side = int(self.num_patches ** 0.5)

        # 8 for 256 patches

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
        Full Global-EBA encoder–decoder backbone. Sign: energetic_mahal_science.py

        Pipeline: (for energetic_mahal_science.py)
        multiscale patch embedding → prepend a CLS token and add
        global positional embeddings → a stack of energy-based
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
            Expose the classifier weight matrix (num_classes × latent_dim)
            as the set of semantic class prototypes, so the prototype-matching
            loss and the classifier share the same parameters.
        """
        return self.classifier.weight

    def encode(self, world):
        """
            Encode an image batch into the latent representation `z`.

            Runs patch embedding, the Transformer stack and the
            prototype cross-attention, then fuses the CLS token
            with the flattened prototype features through the
            encoder head.

            Args:
                world: input images of shape (B, C, H, W).
            Returns:
                Latent tensor `z` of shape (B, latent_dim).
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
            Map a latent vector back to an image reconstruction
            via the convolutional decoder. Returns a tensor of
            shape (B, C, H, W).
        """
        return self.decoder(z_val, encoder_features)

    def classify(self, z_val):
        """
            Produce class logits from a latent vector using
            the linear classifier head.
        """
        return self.classifier(z_val)

    def forward(self, x_val):
        """
            Full forward pass: encode the image into `z`, decode
            it into a reconstruction, and reshape the output to
            (B, 3, img_size, img_size).

            Returns:
                (x_reconstruction, z) — the reconstructed
                image and its latent code.
        """
        zeta, encoder_feats = self.encode(x_val)

        x_reconstruction_flat = self.decode(zeta, encoder_feats)

        x_reconstruction = x_reconstruction_flat.view(-1, 3, self.img_size, self.img_size)

        return x_reconstruction, zeta

    def energy_components(self, sea_1, Y_reconstruction):
        """
            Compute four per-sample reconstruction-discrepancy terms between
            the input and its reconstruction, each reshaped back to image form
            first:
              l1 — mean pixel-wise MSE,
              l2 — absolute difference of total-variation (edge/smoothness mismatch),
              l3 — MSE between log-magnitude 2D FFT spectra (frequency-domain mismatch),
              l4 — mean of locally pooled squared error (coarse spatial mismatch).

            Args:
                sea_1: input images, shape (B, C * H * W) or reshape-compatible.
                Y_reconstruction: reconstructions of the same shape.
            Returns:
                Tensor of shape (B, 4) stacking the four components per sample.
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
        l4 = F.avg_pool2d((x_img - r_img) ** 2, kernel_size=8, stride=4).reshape(B, -1).mean(dim=1)

        return torch.stack([l1, l2, l3, l4], dim=1)

    def compute_loss(self, zeta, Z_reconstruction):
        """
            Training objective: take the four reconstruction components, apply
            the (currently uniform) per-component weights, sum them per sample
            and average over the batch into a single scalar loss.

            Args:
                zeta: input images.
                Z_reconstruction: their reconstructions.
            Returns:
                Scalar reconstruction loss.
        """
        comp = self.energy_components(zeta, Z_reconstruction)

        weights = torch.ones(4, device=comp.device).unsqueeze(0)

        return (weights * comp).sum(dim=1).mean()

    def energy(self, T, T_reconstruction):
        """
            Inference-time anomaly score: the weighted sum of the four reconstruction
            components per sample, computed under `no_grad`. Higher energy indicates a
            poorer reconstruction and thus a more likely anomaly.

            Args:
                T: input images.
                T_reconstruction: their reconstructions.
            Returns:
                Per-sample energy tensor of shape (B,).
        """
        with torch.no_grad():
            comp = self.energy_components(T, T_reconstruction)

            weights = torch.ones(4, device=comp.device).unsqueeze(0)

            return (weights * comp).sum(dim=1)


class FlatImageDataset(Dataset):
    """
        Dataset that handles both tensor and numpy array inputs.
    """

    def __init__(self, extremities, phrases=None, transform_left=None):
        self.images = extremities
        self.labels = phrases

        self.transform = transform_left

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_land = self.images[idx]

        # Convert to tensor if it's not already

        if isinstance(image_land, np.ndarray):
            image_land = torch.from_numpy(image_land).float()
        elif isinstance(image_land, list):
            image_land = torch.tensor(image_land, dtype=torch.float32)

        # Ensure correct shape (C, H, W)

        if image_land.dim() == 1:
            img_size = int(math.sqrt(image_land.shape[0] / 3))

            if img_size * img_size * 3 == image_land.shape[0]:
                image_land = image_land.reshape(3, img_size, img_size)

        if self.transform is not None:
            if isinstance(image_land, torch.Tensor):
                to_pil = ToPILImage()
                image_land = to_pil(image_land)

            image_land = self.transform(image_land)

        if self.labels is None:
            return image_land

        return image_land, self.labels[idx]


def load_dataset_for_anomaly(data_root="./data/brisc2025/brisc2025/classification_task", diff_samples=10000):
    """
        Loads Brain Tumor MRI dataset for one-class anomaly detection.

        Args:
            data_root (str): Path to the dataset folder (train/no_tumor, test/*).
            diff_samples (int): Max number of anomaly samples to include.

        Returns:
            tuple: (train_normal, test_normal, test_anomaly) as lists of torch tensors.
    """

    print(f"Loading Brain Tumor MRI dataset from: {data_root}")

    transform_qed = transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.LANCZOS, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Function to load images_all from a directory

    def load_images_from_dir(directory, transform_fn, avg_sample_numb=None):
        images_all = []

        if not os.path.exists(directory):
            return images_all

        # Look for image files

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
            image_files.extend(glob.glob(os.path.join(directory, ext.upper())))

        # Also check subdirectories

        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            if os.path.isdir(item_path):

                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(item_path, ext)))
                    image_files.extend(glob.glob(os.path.join(item_path, ext.upper())))

        image_files = sorted(set(image_files))

        if avg_sample_numb:
            image_files = image_files[:avg_sample_numb]

        print(f"    Found {len(image_files)} image files in {directory}")

        for img_path in image_files:
            try:
                img_renaissance = Image.open(img_path).convert('RGB')
                img_tensor = transform_fn(img_renaissance)
                images_all.append(img_tensor)

                # Keep as tensor, not flattened

            except Exception as e:
                print(f"      Error loading {img_path}: {e}")
                continue

        return images_all

    # Load training data (normal class only)

    train_normal_images = []
    train_path = os.path.join(data_root, "train")

    if os.path.exists(train_path):
        normal_train_path = os.path.join(train_path, "no_tumor")

        if os.path.exists(normal_train_path):
            train_normal_images = load_images_from_dir(normal_train_path, transform_qed)

            print(f"Loaded {len(train_normal_images)} training normal images_all from {normal_train_path}")
        else:
            # Try to find any subdirectory with images_all

            for subdir in os.listdir(train_path):
                subdir_path = os.path.join(train_path, subdir)

                if os.path.isdir(subdir_path):
                    images_all_v2 = load_images_from_dir(subdir_path, transform_qed)
                    if images_all_v2:
                        train_normal_images.extend(images_all_v2)

                        print(f"Loaded {len(images_all_v2)} training images_all from {subdir_path}")
    else:
        print(f"Warning: Training path {train_path} not found")

    # Load test data

    test_path = os.path.join(data_root, "test")
    test_normal_images = []
    test_anomaly_images = []

    if os.path.exists(test_path):
        anomaly_classes = ['glioma', 'meningioma', 'pituitary']

        for class_name in os.listdir(test_path):
            class_path = os.path.join(test_path, class_name)

            if os.path.isdir(class_path):
                images_all_v2 = load_images_from_dir(class_path, transform_qed)

                if class_name.lower() == 'no_tumor':
                    test_normal_images.extend(images_all_v2)

                    print(f"  Loaded {len(images_all_v2)} normal (no_tumor) images_all")
                elif class_name.lower() in anomaly_classes:
                    test_anomaly_images.extend(images_all_v2)

                    print(f"  Loaded {len(images_all_v2)} anomaly ({class_name}) images_all")

    # Apply sample limit

    if diff_samples is not None and len(test_anomaly_images) > diff_samples:
        test_anomaly_images = test_anomaly_images[:diff_samples]

        print(f"Limited anomaly samples to {diff_samples}")

    print(f"\nFinal dataset statistics:")
    print(f"Train normal (no_tumor): {len(train_normal_images)}")
    print(f"Test normal (no_tumor): {len(test_normal_images)}")
    print(f"Test anomaly (glioma/meningioma/pituitary): {len(test_anomaly_images)}")

    # Verify we have tensors

    if train_normal_images and isinstance(train_normal_images[0], torch.Tensor):
        print(f"Sample tensor shape: {train_normal_images[0].shape}")

    return train_normal_images, test_normal_images, test_anomaly_images


@torch.no_grad()
def compute_mahalanobis_stats(model_second, loader, device):
    """
        Estimate the latent-space Gaussian statistics of the normal class.

        Encodes every sample in the loader, computes the empirical mean and a
        Ledoit-Wolf shrinkage covariance, adds diagonal regularization if the
        covariance is severely ill-conditioned (condition number > 1e6), and
        inverts it.

        Args:
            model_second: SELYNE model used as the feature encoder.
            loader: DataLoader of normal samples.
            device: torch device.
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
        Compute the squared Mahalanobis distance of each latent vector from
        the class mean under the given inverse covariance: (z − μ)ᵀ Σ⁻¹ (z − μ).

        Args:
            zero_3: latent vectors of shape (B, latent_dim).
            mean: class mean of shape (latent_dim,).
            cov_inv: inverse covariance of shape (latent_dim, latent_dim).
        Returns:
            Per-sample distance tensor of shape (B,).
    """
    delta = zero_3 - mean.unsqueeze(0)

    return torch.einsum('bi,ij,bj->b', delta, cov_inv, delta)


@torch.no_grad()
def evaluate_mahalanobis(model_three, normal_loader, anomaly_loader, device, mean, cov_inv):
    """
        Args:
            model_three: SELYNE encoder.
            normal_loader: DataLoader for the normal class.
            anomaly_loader: DataLoader for the anomaly class.
            device: torch device.
            mean: latent class mean from compute_mahalanobis_stats.
            cov_inv: latent inverse covariance from compute_mahalanobis_stats.
        Returns:
            Dict with auroc, fpr, tpr, and the
            raw normal/anomaly distance arrays.
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
def evaluate_energy(model_forth, normal_loader, anomaly_loader, device):
    """
        Args:
            model_forth: SELYNE model.
            normal_loader: DataLoader for the normal class.
            anomaly_loader: DataLoader for the anomaly class.
            device: torch device.
        Returns:
            Dict with auroc, fpr, tpr, and the
            raw normal/anomaly energy arrays.
            :param model_forth:
            :param normal_loader:
            :param anomaly_loader:
            :param device:
    """
    model_forth.eval()

    def collect_energies(loader):
        energies = []

        for x_lot_3 in loader:
            x_lot_3 = x_lot_3.to(device)
            recon, _ = model_forth(x_lot_3)

            e_value = model_forth.energy(x_lot_3, recon)

            energies.extend(e_value.cpu().numpy().tolist())

        return np.asarray(energies, dtype=float)

    e_normal = collect_energies(normal_loader)
    e_anomaly = collect_energies(anomaly_loader)

    y_true = np.array([0] * len(e_normal) + [1] * len(e_anomaly))
    scores = np.concatenate([e_normal, e_anomaly])

    auroc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    print(f"ENERGY -> AUROC: {auroc:.4f} | "
          f"Normal mean energy: {np.mean(e_normal):.4f} | Anomaly mean energy: {np.mean(e_anomaly):.4f}")

    return {'auroc': float(auroc),
            'fpr': fpr,
            'tpr': tpr,
            'normal': e_normal,
            'anomaly': e_anomaly}


def augment_batch(x, img_size, in_chans, pad=2):
    """
        Light, tensor-based augmentation to produce two views.
    """
    B = x.shape[0]

    ImageMeetHall = x.view(B, in_chans, img_size, img_size)

    flip = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1, 1)
    ImageMeetHall = torch.where(flip, ImageMeetHall.flip(-1), ImageMeetHall)

    sh, sw = random.randint(-2, 2), random.randint(-2, 2)

    ImageMeetHall = F.pad(ImageMeetHall, (pad, pad, pad, pad), mode='reflect')

    ImageMeetHall = ImageMeetHall[:, :, pad + sh: pad + sh + img_size, pad + sw: pad + sw + img_size]
    ImageMeetHall = ImageMeetHall + 0.001 * torch.randn_like(ImageMeetHall)

    return ImageMeetHall


@torch.no_grad()
def ema_update(student, teacher, decay):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(decay).add_(ps.data, alpha=1 - decay)

    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


def latent_consistency(z_s, z_t, var_gamma=0.5):
    """
        BYOL invariance (cosine distance) + VICReg
        variance hinge (per-dimension std dev penalty).
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
            model_five: pretrained SELYNE model.
            train_loader_class: DataLoader of the single normal class.
            epochs, lr, wd, ema_decay: optimisation hyperparameters.
            device: torch device.
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
            # (energy) + latent

            recon, z_s = model_five(v1)
            energy_loss = model_five.compute_loss(v1, recon)

            # teacher (frozen, momentum):
            # view2 -> latent target

            with torch.no_grad():
                z_t, _ = teacher.encode(v2)

            inv_loss, var_loss = latent_consistency(z_s, z_t)
            consist_loss = inv_loss + var_loss

            loss_input = K_o * energy_loss + (math.pi / 10) * consist_loss

            optimizer.zero_grad()
            loss_input.backward()

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optimizer.step()

            # momentum update: teacher's
            # knowledge slowly follows the
            # student

            ema_update(model_five, teacher, ema_decay)

            tot_e += energy_loss.item()
            tot_c += consist_loss.item()

            tot_std += z_s.detach().std(dim=0).mean().item()

            n += 1

        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch:3d} | Energy: {tot_e / n:.4f} | Consist: {tot_c / n:.4f} | "
              f"LatentStd: {tot_std / n:.4f} | LR: {lr_now:.2e}")

    return model_five


@torch.no_grad()
def save_reconstructions(model_sixth, loader, device, out_dir="reconstructions",
                         max_samples=20, show_metrics=True):
    """
    Reconstruct images from a loader and save original | reconstruction
    pairs as PNGs with quality metrics.
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

    param_seed_value = 1
    set_seed(param_seed_value)

    print(f"Device: {DEVICE}")

    # Load Brain Tumor MRI
    # dataset (single class)

    train_n, test_n, test_a = load_dataset_for_anomaly(data_root="./data/brisc2025/brisc2025/classification_task")

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

    checkpoint_path = "pretrained_selyne_full_recon_best.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Warning: {checkpoint_path} not found. Using random init.")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint, strict=True)
        print("Pretrained model loaded.")

    print("\nFine-tuning with Mahalanobis enhancement")

    model = train_mahalanobis_enhanced(model, train_loader, DEVICE, epochs=30, lr=1e-4)

    mean_lat, cov_inv_lat = compute_mahalanobis_stats(model, train_loader, DEVICE)

    print("\nEnergy-Based Detection")

    energy_results = evaluate_energy(model, test_normal_loader, test_anomaly_loader, DEVICE)

    print("\nMahalanobis Distance Detection")

    mahalanobis_results = evaluate_mahalanobis(model, test_normal_loader, test_anomaly_loader, DEVICE, mean_lat,
                                               cov_inv_lat)

    print("FINAL RESULTS (Brain Tumor MRI)")
    print(f"Energy AUROC: {energy_results['auroc']:.4f}")

    print(f"Mahalanobis AUROC: {mahalanobis_results['auroc']:.4f}")

    torch.save({
        'encoder_head': model.encoder_head.state_dict(),
        'proto_attn': model.proto_attn.state_dict(),
        'mean': mean_lat.cpu(),
        'cov_inv': cov_inv_lat.cpu(),
    }, "selyne_brain_tumor.pt")

    print(f"\n✓ Model saved")

    save_reconstructions(model, test_normal_loader, DEVICE, out_dir="recon_brain_normal", show_metrics=True)
    save_reconstructions(model, test_anomaly_loader, DEVICE, out_dir="recon_brain_anomaly", show_metrics=True)

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