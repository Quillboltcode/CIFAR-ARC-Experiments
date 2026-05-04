from typing import Optional, Tuple

from utils.pos_embed import VisionRotaryEmbeddingFast
import torch
from torch import nn

from timm.models.vision_transformer import PatchEmbed


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with RoPE."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require the head dimension to be even")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2
        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=int(max_seq_len ** 0.5),
            no_rope=no_rope,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                mask,
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context


class CIFARTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for CIFAR models."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 0,
        no_rope: int = 0,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = residual + self.dropout3(x)
        x = self.norm2(x)
        return x


class CIFARTransformerEncoder(nn.Module):
    """Transformer encoder stack."""
    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CIFARTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    no_rope=no_rope,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class CIFARViT(nn.Module):
    """Vision Transformer for CIFAR image classification.
    
    Standard ViT architecture with adaptations for 32x32 RGB images.
    Features:
    - Learned CLS token for global image representation
    - Standard patch embedding for 3-channel input
    - Per-patch learned positional embeddings
    - Classification head with mean pooling option
    """
    
    def __init__(
        self,
        image_size: int = 32,
        num_classes: int = 10,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        dropout: float = 0.0,
        patch_size: int = 4,
        use_cls_token: bool = True,
        use_mean_pooling: bool = False,
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("`image_size` must be > 0.")
        if num_classes <= 0:
            raise ValueError("`num_classes` must be > 0.")
        if embed_dim <= 0:
            raise ValueError("`embed_dim` must be > 0.")

        self.image_size = image_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.use_cls_token = use_cls_token
        self.use_mean_pooling = use_mean_pooling

        # Patch embedding: 3 RGB channels -> embed_dim
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            bias=True
        )

        # CLS token (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # Positional embeddings for all tokens (CLS + patches if using CLS, else just patches)
        seq_len = self.num_patches + (1 if use_cls_token else 0)
        self.positional_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # no_rope: exclude CLS token from rotary if using CLS
        no_rope = 1 if use_cls_token else 0

        self.encoder = CIFARTransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=seq_len,
            no_rope=no_rope,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) RGB images normalized to [0,1] or with ImageNet stats
            attention_mask: Optional (B, H, W) mask, 1 for valid, 0 for padded
        Returns:
            logits: (B, num_classes) classification logits
        """
        if pixel_values.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {pixel_values.shape}")
        if pixel_values.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {pixel_values.size(1)}")
        if pixel_values.size(2) != self.image_size or pixel_values.size(3) != self.image_size:
            raise ValueError(
                f"Input image size must be {self.image_size}x{self.image_size}, "
                f"got {pixel_values.size(2)}x{pixel_values.size(3)}"
            )

        batch_size = pixel_values.size(0)

        # Patch embedding: (B, 3, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(pixel_values)

        # Prepend CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+num_patches, embed_dim)

        # Add positional embeddings
        x = x + self.positional_embed
        x = self.dropout(x)

        # Prepare attention mask if provided (only affects patch tokens, not CLS)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, self.image_size, self.image_size):
                raise ValueError(f"attention_mask shape must be (B, {self.image_size}, {self.image_size})")
            # Downsample mask to patch level
            if self.patch_size > 1:
                attention_mask = attention_mask.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
                attention_mask = attention_mask.mean(dim=(2, 3))  # Average over patch
                attention_mask = (attention_mask > 0.5).float()
            else:
                attention_mask = attention_mask.float()
            flat_mask = attention_mask.view(batch_size, -1)
            if self.use_cls_token:
                pad_mask = ~flat_mask.bool()
                pad_mask = torch.cat(
                    [torch.zeros(batch_size, 1, device=pixel_values.device, dtype=torch.bool), pad_mask],
                    dim=1
                )
                key_padding_mask = pad_mask
            else:
                key_padding_mask = ~flat_mask.bool()

        # Transformer encoder
        x = self.encoder(x, key_padding_mask=key_padding_mask)

        # Normalize
        x = self.norm(x)

        # Global pooling
        if self.use_cls_token:
            x = x[:, 0]  # Use CLS token representation
        elif self.use_mean_pooling:
            x = x.mean(dim=1)  # Mean over all patches
        else:
            # Default: use first token (equivalent to CLS token without explicit CLS)
            x = x[:, 0]

        # Classification head
        logits = self.head(x)
        return logits
