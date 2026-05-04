from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed

# Import standard ViT layer (LayerNorm + MLP)
from src.ARC_ViT import ARCTransformerEncoderLayer as StandardLayer


@dataclass
class LoopForwardMetadata:
    """Metadata for looped forward pass."""
    gate_probs: List[torch.Tensor]
    exit_steps: torch.Tensor
    max_steps: int


class CIFARLoopViT(nn.Module):
    """Looped Vision Transformer for CIFAR classification.
    
    Core ideas:
    - Shared transformer layers applied repeatedly (feature recycling)
    - Optional early exit via learnable gate
    - Optional step embeddings to distinguish iterations
    - CLS token for global image representation
    """
    
    def __init__(
        self,
        image_size: int = 32,
        num_classes: int = 10,
        embed_dim: int = 384,
        loop_core_depth: int = 2,
        max_loop_steps: int = 6,
        min_loop_steps: int = 1,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        dropout: float = 0.0,
        patch_size: int = 4,
        use_cls_token: bool = True,
        use_exit_gate: bool = False,
        gate_threshold: float = 0.5,
        add_step_embeddings: bool = True,
    ) -> None:
        super().__init__()

        # Validation
        if image_size <= 0: raise ValueError("image_size must be > 0")
        if embed_dim <= 0: raise ValueError("embed_dim must be > 0")
        if loop_core_depth <= 0: raise ValueError("loop_core_depth must be > 0")
        if max_loop_steps <= 0: raise ValueError("max_loop_steps must be > 0")
        if min_loop_steps < 1 or min_loop_steps > max_loop_steps:
            raise ValueError("min_loop_steps must be in [1, max_loop_steps]")

        self.image_size = image_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.loop_core_depth = loop_core_depth
        self.max_loop_steps = max_loop_steps
        self.min_loop_steps = min_loop_steps
        self.use_exit_gate = use_exit_gate
        self.gate_threshold = gate_threshold
        self.add_step_embeddings = add_step_embeddings
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.use_cls_token = use_cls_token

        # Patch embedding (3 RGB channels)
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            bias=True,
        )

        # CLS token for classification
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        seq_len = self.num_patches + (1 if use_cls_token else 0)
        # Positional embeddings for all tokens (CLS + patches if using CLS)
        seq_len = self.num_patches + (1 if use_cls_token else 0)
        self.positional_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Shared loop layers: exclude CLS token from rotary if present
        layer_no_rope = 1 if use_cls_token else 0
        self.core_layers = nn.ModuleList([
            StandardLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                max_seq_len=seq_len,
                no_rope=layer_no_rope,
            )
            for _ in range(loop_core_depth)
        ])

        # Optional step embeddings
        if add_step_embeddings:
            self.step_embed = nn.Parameter(torch.zeros(max_loop_steps, embed_dim))
        else:
            self.step_embed = None

        # Exit gate (for early exit decision)
        if use_exit_gate:
            self.exit_gate = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
            )
        else:
            self.exit_gate = None

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.step_embed is not None:
            nn.init.trunc_normal_(self.step_embed, std=0.02)
        if self.exit_gate is not None:
            for m in self.exit_gate:
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dynamic_exit: bool = False,
        gate_threshold: Optional[float] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, LoopForwardMetadata] | Tuple[torch.Tensor, LoopForwardMetadata, List[torch.Tensor]]:
        """
        Args:
            pixel_values: (B, 3, H, W) RGB images
            attention_mask: Optional (B, H, W) binary mask
            dynamic_exit: If True, use exit gate to stop early
            gate_threshold: Threshold for exit gate
            return_intermediates: If True, also return logits from all steps
        Returns:
            logits: (B, num_classes)
            metadata: LoopForwardMetadata with gate probs and exit steps
            intermediate_logits: optional list of per-step logits
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # Patch embedding
        x = self.patch_embed(pixel_values)  # (B, num_patches, embed_dim)

        # Prepend CLS token if used (must come before positional embedding)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)  # (B, seq_len, embed_dim)

        # Add positional embeddings to all tokens
        x = x + self.positional_embed
        x = self.dropout(x)

        # Attention mask
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, self.image_size, self.image_size):
                raise ValueError(f"attention_mask shape must be (B, {self.image_size}, {self.image_size})")
            if self.patch_size > 1:
                attention_mask = attention_mask.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
                attention_mask = attention_mask.mean(dim=(2, 3))
                attention_mask = (attention_mask > 0.5).float()
            else:
                attention_mask = attention_mask.float()
            flat_mask = attention_mask.view(batch_size, -1)
            pad_mask = ~flat_mask.bool()
            if self.use_cls_token:
                pad_mask = torch.cat([torch.zeros(batch_size, 1, device=device, dtype=torch.bool), pad_mask], dim=1)
            key_padding_mask = pad_mask

        # Looped processing
        running_hidden = x
        initial_hidden = x
        device = running_hidden.device
        threshold = gate_threshold if gate_threshold is not None else self.gate_threshold

        if dynamic_exit and self.use_exit_gate:
            finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            cached_final = torch.zeros_like(running_hidden)
            exit_steps = torch.full((batch_size,), self.max_loop_steps, dtype=torch.long, device=device)
        else:
            finished_mask = None
            cached_final = None
            exit_steps = torch.full((batch_size,), self.max_loop_steps, dtype=torch.long, device=device)

        gate_probs: List[torch.Tensor] = []
        intermediate_logits_list: List[torch.Tensor] = []

        for step in range(self.max_loop_steps):
            # Step embedding
            if self.step_embed is not None:
                step_emb = self.step_embed[min(step, self.step_embed.size(0) - 1)]
                running_hidden = running_hidden + step_emb.view(1, 1, -1)

            # Shared layers
            for layer in self.core_layers:
                running_hidden = layer(running_hidden, key_padding_mask=key_padding_mask)

            if return_intermediates:
                step_logits = self._classify(running_hidden)
                intermediate_logits_list.append(step_logits)

            # Exit gate
            if self.use_exit_gate:
                gate_logit = self.exit_gate(running_hidden[:, 0, :]).squeeze(-1)
                gate_prob = torch.sigmoid(gate_logit)
                gate_probs.append(gate_prob)
            else:
                gate_prob = None

            # Dynamic exit check
            if dynamic_exit and self.use_exit_gate and gate_prob is not None:
                eligible = (step + 1) >= self.min_loop_steps
                exit_now = (gate_prob >= threshold) & eligible & (~finished_mask)
                if exit_now.any():
                    cached_final[exit_now] = running_hidden[exit_now]
                    exit_steps[exit_now] = step + 1
                    finished_mask = finished_mask | exit_now
                # Replace hidden states for finished samples with cached
                running_hidden = torch.where(
                    finished_mask.view(batch_size, 1, 1),
                    cached_final,
                    running_hidden,
                )
                if finished_mask.all():
                    break

        # Final representation
        final_states = self.norm(running_hidden)
        logits = self._classify(final_states)

        metadata = LoopForwardMetadata(
            gate_probs=gate_probs,
            exit_steps=exit_steps,
            max_steps=self.max_loop_steps,
        )

        if return_intermediates:
            return logits, metadata, intermediate_logits_list
        return logits, metadata

    def _classify(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract classification token and project to logits."""
        if self.use_cls_token:
            cls_repr = hidden_states[:, 0]
        else:
            cls_repr = hidden_states.mean(dim=1)
        return self.head(cls_repr)
