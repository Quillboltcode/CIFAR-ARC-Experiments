from typing import Optional, Tuple
import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed
from utils.pos_embed import VisionRotaryEmbeddingFast

# Import RMSNorm and ConvolutionalGLU from ViT1
from src.ARC_ViT1 import RMSNorm, ConvolutionalGLU


class RelativePositionBias(nn.Module):
    """2D Relative Positional Bias for vision transformers.
    
    Learnable bias table based on relative distance between patches.
    Supports dynamic grid sizes via on-the-fly index regeneration.
    """
    def __init__(self, num_heads: int, grid_size: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.grid_size = grid_size
        # Table size: (2*grid_size-1) * (2*grid_size-1) possible relative positions
        self.num_relative_distance = (2 * grid_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )
        # Generate relative position index for the default square size
        self.register_buffer("relative_position_index", self._generate_relative_position_index(grid_size, grid_size))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _generate_relative_position_index(self, h: int, w: int) -> torch.Tensor:
        """Generate relative position index for a grid of shape (h, w).
        
        For square grids, h == w == grid_size.
        """
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, H, W
        coords_flatten = coords.view(2, -1)  # 2, H*W
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2

        # Shift to start from 0 based on the initialized base size (self.grid_size)
        relative_coords[:, :, 0] += self.grid_size - 1
        relative_coords[:, :, 1] += self.grid_size - 1

        # Clamp to table bounds
        max_offset = 2 * self.grid_size - 2
        relative_coords[:, :, 0] = relative_coords[:, :, 0].clamp(0, max_offset)
        relative_coords[:, :, 1] = relative_coords[:, :, 1].clamp(0, max_offset)

        relative_coords[:, :, 0] *= 2 * self.grid_size - 1
        relative_position_index = relative_coords.sum(-1)  # H*W, H*W
        return relative_position_index

    def forward(self, h: int, w: int) -> torch.Tensor:
        """
        Args:
            h: Current grid height (number of patches)
            w: Current grid width (number of patches)
        Returns:
            bias: (1, num_heads, H*W, H*W)
        """
        # For CIFAR we assume square grids, but support rectangle for generality
        if h == self.grid_size and w == self.grid_size:
            index = self.relative_position_index
        else:
            # Generate on-the-fly for different sizes
            device = self.relative_position_bias_table.device
            index = self._generate_relative_position_index(h, w).to(device)

        bias = self.relative_position_bias_table[index.view(-1)]
        bias = bias.view(h * w, h * w, -1)
        bias = bias.permute(2, 0, 1).contiguous()
        return bias.unsqueeze(0)


class MultiHeadSelfAttentionWithBias(nn.Module):
    """Multi-head attention with RoPE and relative positional bias.
    
    Supports excluding some initial tokens (e.g., CLS or task tokens) from
    RoPE and relative bias via the `no_rope` parameter.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 0,
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.no_rope = no_rope

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2

        # RoPE setup: for image tokens only (exclude first no_rope tokens)
        img_seq_len = max_seq_len - no_rope
        if img_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than no_rope")
        rope_grid_size = int(img_seq_len ** 0.5)
        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=rope_grid_size,
            no_rope=no_rope,
        )

        # Relative positional bias (for image tokens only)
        self.rel_pos_bias = RelativePositionBias(num_heads, grid_size)

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

        # Inject relative position bias only for image token region
        n_task = self.no_rope
        num_img_tokens = seq_len - n_task
        # Current spatial dimensions (assumes square)
        grid_sz = int(num_img_tokens ** 0.5)
        bias = self.rel_pos_bias(grid_sz, grid_sz)
        attn_scores[:, :, n_task:, n_task:] = attn_scores[:, :, n_task:, n_task:] + bias

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context


class CIFARViT2TransformerEncoderLayer(nn.Module):
    """Transformer layer with RMSNorm + ConvolutionalGLU + relative bias."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 0,
        no_rope: int = 0,
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.num_task_tokens = no_rope

        self.self_attn = MultiHeadSelfAttentionWithBias(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
            grid_size=grid_size,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(embed_dim)

        # ConvolutionalGLU with no task tokens (image tokens only)
        self.mlp = ConvolutionalGLU(
            in_features=embed_dim,
            hidden_features=mlp_dim,
            act_layer=nn.GELU,
            drop=dropout,
            num_task_tokens=no_rope,
        )

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)

        residual = x
        x = self.norm2(x)
        seq_len = x.shape[1]
        num_img_tokens = seq_len - self.num_task_tokens
        grid_size = int(num_img_tokens ** 0.5)
        x = self.mlp(x, H=grid_size, W=grid_size)
        x = residual + self.dropout2(x)
        return x


class CIFARViT2Encoder(nn.Module):
    """Transformer encoder with ViT2 layers."""
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
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CIFARViT2TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                max_seq_len=max_seq_len,
                no_rope=no_rope,
                grid_size=grid_size,
            )
            for _ in range(depth)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class CIFARViT2(nn.Module):
    """Enhanced ViT with relative positional bias and advanced norms/MLP for CIFAR.
    
    Architecture innovations:
    - RMSNorm instead of LayerNorm
    - ConvolutionalGLU with depthwise conv for spatial inductive bias
    - Relative positional bias combined with RoPE
    - Standard classification head with CLS token
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
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("`image_size` must be > 0.")
        if num_classes <= 0:
            raise ValueError("`num_classes` must be > 0.")

        self.image_size = image_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.use_cls_token = use_cls_token
        self.grid_size = image_size // patch_size

        # Patch embedding: RGB (3 channels) -> embed_dim
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            bias=True
        )

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        seq_len = self.num_patches + (1 if use_cls_token else 0)
        self.no_rope = 1 if use_cls_token else 0  # number of tokens to exclude from RoPE/bias

        # Positional embeddings (for all tokens: CLS + patches)
        self.positional_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.encoder = CIFARViT2Encoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=seq_len,
            no_rope=self.no_rope,
            grid_size=self.grid_size,
        )

        self.norm = RMSNorm(embed_dim)
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
            pixel_values: (B, 3, H, W) RGB images
            attention_mask: Optional (B, H, W) mask
        Returns:
            logits: (B, num_classes)
        """
        if pixel_values.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {pixel_values.shape}")
        if pixel_values.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {pixel_values.size(1)}")

        batch_size = pixel_values.size(0)

        # Patch embedding -> (B, num_patches, embed_dim)
        x = self.patch_embed(pixel_values)

        # Prepend CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.positional_embed
        x = self.dropout(x)

        # Attention mask processing (optional)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, self.image_size, self.image_size):
                raise ValueError(f"attention_mask shape mismatch")
            # Downsample mask from image space to patch space
            if self.patch_size > 1:
                # Unfold to get patch-level average
                attention_mask = attention_mask.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
                attention_mask = attention_mask.mean(dim=(2, 3))
                attention_mask = (attention_mask > 0.5).float()
            else:
                attention_mask = attention_mask.float()
            flat_mask = attention_mask.view(batch_size, -1)
            pad_mask = ~flat_mask.bool()
            if self.use_cls_token:
                pad_mask = torch.cat([torch.zeros(batch_size, 1, device=pixel_values.device, dtype=torch.bool), pad_mask], dim=1)
            key_padding_mask = pad_mask

        # Transformer
        x = self.encoder(x, key_padding_mask=key_padding_mask)

        # Final norm
        x = self.norm(x)

        # Global pooling
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        # Classification
        logits = self.head(x)
        return logits
