# CIFAR Adaptation Changelog

## Overview
This document records all changes made to adapt the ARC-focused Vision Transformer and LoopViT models to work with CIFAR-10/CIFAR-100 datasets.

**Reference Plan:** `.kilo/plans/1777708273669-nimble-otter.md`

---

## Phase 1: Core Architectural Changes (Applied to All Models)

### 1.1 Input Representation
| Aspect | ARC Models | CIFAR Models | Change |
|--------|-----------|--------------|--------|
| Input type | Discrete color indices (0-9) | Continuous RGB pixels (3 channels, [0,1] normalized) | Replaced `color_embed = nn.Embedding(num_colors, embed_dim)` with `PatchEmbed(in_chans=3)` |
| Input shape | `(B, H, W)` integer grid | `(B, 3, H, W)` float tensor | Updated validation in `forward()` |
| Preprocessing | `color_embed(pixel_values.long())` | Direct patch embedding via `PatchEmbed` | Removed discrete embedding lookup |

### 1.2 Task Token Removal
| Aspect | ARC Models | CIFAR Models | Change |
|--------|-----------|--------------|--------|
| Task tokens | Yes (`num_task_tokens=1`, learnable `task_token_embed`) | No (`num_task_tokens=0`) | Removed `task_token_embed` module |
| Sequence composition | `[task_tokens, patch_tokens]` | `[patch_tokens]` or `[CLS, patch_tokens]` | Removed task token concatenation in `forward()` |
| `no_rope` parameter | Typically `1` (exclude task token from RoPE) | `0` or `1` (exclude CLS if present) | Updated RoPE exclusion logic |

### 1.3 Output Head
| Aspect | ARC Models | CIFAR Models | Change |
|--------|-----------|--------------|--------|
| Task | Per-pixel color prediction (dense) | Image-level classification (global) | Replaced pixel-wise head with global classification head |
| Output dimension | `num_colors * patch_area` | `num_classes` (10 or 100) | Changed `self.head = nn.Linear(embed_dim, num_classes)` |
| Output reshaping | Complex reshape to `(B, C, H, W)` | Direct logits `(B, num_classes)` | Removed per-pixel reshape logic |
| Pooling | Uses all patch tokens | CLS token or mean pooling | Added `_classify()` method with CLS/mean options |

### 1.4 Positional Embeddings
| Aspect | ARC Models | CIFAR Models | Change |
|--------|-----------|--------------|--------|
| CLS token | No CLS (uses task token for context) | Optional CLS token | Added optional `self.cls_token` |
| Positional length | `seq_length` (patches only) | `num_patches + 1` if CLS else `num_patches` | Updated `seq_len` calculation |
| Positional tensor shape | `(1, seq_length, embed_dim)` | `(1, seq_len, embed_dim)` matching full sequence | No fundamental change, but seq_len now includes CLS if used |
| Addition order | `tokens + positional_embed` before task token concat | CLS added before or after positional? **FIXED**: CLS prepended first, then add positional (standard ViT) | Ensures CLS gets its own positional embedding at index 0 |

---

## Phase 2: Model-Specific Adaptations

### 2.1 CIFAR_ViT (Standard ViT)
**Source:** `src/ARC_ViT.py`

**Key modifications:**
- Removed `color_embed`, `task_token_embed`
- Added optional `cls_token` parameter
- Changed `self.head` output to `num_classes`
- Added `_classify()` for global pooling (CLS or mean)
- Updated `CIFARTransformerEncoderLayer` to accept `no_rope` parameter (passed through encoder)
- Set `no_rope = 1` if `use_cls_token=True`, else `0`
- Attention mask handling: downsample image mask to patch level using `unfold` + mean
- **New parameters:** `use_cls_token` (default True), `use_mean_pooling` (default False)

**Files created:**
- `src/CIFAR_ViT.py` (~316 lines)

---

### 2.2 CIFAR_ViT2 (Enhanced with Relative Bias)
**Source:** `src/ARC_ViT2.py` + `src/ARC_ViT1.py` components (RMSNorm, ConvolutionalGLU)

**Key modifications:**
- Reused `RMSNorm` and `ConvolutionalGLU` from `ARC_ViT1.py` (unchanged)
- `RelativePositionBias` class:
  - Changed `forward()` signature from `forward(self, grid_size)` to `forward(self, h, w)` to support non-square grids
  - `_generate_relative_position_index()` now takes `h` and `w` separately
  - Bias table indexing uses base `self.grid_size` (initialized size) for clamping
- `MultiHeadSelfAttentionWithBias`:
  - Added `no_rope` parameter to constructor (passed to `VisionRotaryEmbeddingFast`)
  - Bias injection only on image token region: `attn_scores[:, :, no_rope:, no_rope:] += bias`
  - Computes dynamic `grid_sz` from `(seq_len - no_rope)` and calls `self.rel_pos_bias(grid_sz, grid_sz)`
- `CIFARViT2TransformerEncoderLayer`:
  - Accepts `no_rope` and passes to attention and `ConvolutionalGLU`
  - `forward()` computes `num_img_tokens = seq_len - self.num_task_tokens` before MLP
- `CIFARViT2`:
  - Optional `use_cls_token` → sets `self.no_rope = 1` if CLS used else `0`
  - Positional embedding shape includes CLS if present
  - CLS token prepended **before** adding positional embeddings
  - Passes `max_seq_len` and `no_rope` to encoder

**Files created:**
- `src/CIFAR_ViT2.py` (~406 lines)

**Bug fixes during adaptation:**
- Fixed `RelativePositionBias._generate_relative_position_index()`: originally took single `size` param; corrected to accept `(h, w)` and handle rectangular (though CIFAR patches are square)
- Fixed `CIFARViT2Encoder.__init__`: had duplicate list comprehension (lines 229-245) due to edit error — replaced with single generator expression

---

### 2.3 CIFAR_LoopViT (Looped Architecture)
**Source:** `src/ARC_LoopViT.py`

**Key modifications:**
- Removed `color_embed`, `task_token_embed`
- Added optional `cls_token`
- Changed `self.head` to `num_classes`
- Removed `_project_to_logits()`; added `_classify()` with CLS/mean pooling
- Updated `forward()`:
  - No per-pixel reshape at output
  - `_classify()` extracts `hidden_states[:, 0]` or `mean(dim=1)`
- `core_layers` built with `StandardLayer` (from ARC_ViT) with `no_rope=1` if CLS used else `0`
- Fixed positional embedding order: CLS prepended **before** adding positional embeddings
- Dynamic exit logic: Uses `finished_mask` boolean tensor to track exited samples; supports `return_intermediates`
- **Removed**: `_encode_inputs()`, `_prepare_attention_mask()` (simplified inline in `forward()`)
- **Removed**: `_project_to_logits()` (ARC-specific pixel reshaping)

**Files created:**
- `src/CIFAR_LoopViT.py` (~272 lines)

**Bug fixes during adaptation:**
- Fixed `positional_embed` shape to include CLS token (`seq_len = num_patches + 1`)
- Fixed layer instantiation to pass `no_rope` to respect CLS exclusion from RoPE
- Fixed dynamic exit condition: `exit_now = (gate_prob >= threshold) & eligible & (~finished_mask)` (previously had incorrect `(~exit_steps.lt(step+1).any())`)
- Fixed `running_hidden` replacement: `torch.where(finished_mask.view(...), cached_final, running_hidden)` properly caches per-sample final states

---

### 2.4 CIFAR_LoopViT_InputInject (Input-Injection Variant)
**Source:** `src/ARC_LoopViT_InputInject.py`

**Key modifications:**
- Mirrors `CIFAR_LoopViT` changes (CLS token, classification head, no task tokens)
- Preserves input injection core: `running_hidden = running_hidden + initial_hidden` at each iteration
- Fixed positional embedding ordering and `no_rope` passing (same as LoopViT)
- Fixed dynamic exit logic to use `finished_mask` properly
- Step embeddings use `self.step_embed` (learnable `nn.Parameter`) — fixed `.weight` access

**Files created:**
- `src/CIFAR_LoopViT_InputInject.py` (~259 lines)

**Bug fixes:**
- Changed `self.step_embed.weight[step]` to `self.step_embed[min(step, self.step_embed.size(0)-1)]` since it's `nn.Parameter`, not `nn.Embedding`
- Fixed duplicate `for` comprehensions in `__init__` (not an issue here; code clean)

---

## Phase 3: New Infrastructure Files

### 3.1 Data Loading — `src/CIFAR_loader.py`
**New file** (~388 lines total)

**Key components:**
- `CIFARDataset`: Wrapper around `torchvision.datasets.CIFAR10/100`
- `get_cifar_transforms()`: Returns `transforms.Compose` with:
  - Train: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize(mean, std)`
  - Test: `ToTensor()`, `Normalize(mean, std)`
- Constants: `CIFAR10_MEAN/STD`, `CIFAR100_MEAN/STD`
- `build_dataloaders()`: Creates train/val loaders with optional distributed sampler
  - Returns: `(train_loader, val_loader, train_dataset, val_dataset)`
- `collate_fn_cifar()`: Stacks `pixel_values` and `labels` into batch dict

**Differences from `ARC_loader.py`:**
- No task IDs (single dataset)
- No attention masks needed (images fully observed)
- No resolution/translation augmentations (standard CIFAR augmentations only)
- Simpler collate: just images + labels

---

### 3.2 Training — `src/CIFAR_trainer.py`
**New file** (~416 lines)

**Key components:**
- `parse_args()`: Extensive CLI arguments for model, training, optimizer, execution
- `build_model()`: Factory returning appropriate model based on `args.model`
- `setup_optimizer_and_scheduler()`:
  - AdamW with separate weight decay for bias/norm/pos embed (no decay) vs weights
  - Warmup (`LinearLR`) + cosine decay (`CosineAnnealingLR`) via `SequentialLR`
- `train_epoch()`: Standard training loop with optional AMP
- `evaluate()`: Validation loop
- `save_checkpoint()` / `load_checkpoint()`: Persistence
- `main()`: End-to-end pipeline

**Differences from ARC training:**
- Single-label classification loss (`CrossEntropyLoss`) vs per-pixel cross-entropy
- No task IDs passed to model
- No complex attention masks (all images fully visible)
- Simpler metric: top-1 accuracy only
- Checkpoint includes `args` for config reconstruction

---

### 3.3 Evaluation — `src/CIFAR_evaluator.py`
**New file** (~180 lines)

**Key functions:**
- `load_checkpoint_and_build_model()`: Infers architecture from saved `args`
- `evaluate()`: Computes loss and accuracy on test set
- CLI interface for quick evaluation

---

### 3.4 Requirements — `requirements.txt`
**New file**

**Core dependencies:**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
einops>=0.7.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=9.0.0
tqdm>=4.62.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
tensorboard>=2.10.0
```

**Note:** Pinned to versions compatible with both local (Python 3.12) and Kaggle (Python 3.9+) environments.

---

### 3.5 Execution Scripts
- **`run_training.sh`**: Bash wrapper calling `python -m src.CIFAR_trainer` with arguments
- **`kaggle_notebook.ipynb`**: Jupyter notebook with configuration, dataset download, model build, and training cells ready for Kaggle

---

## Phase 4: Bug Fixes & Compatibility Adjustments

### 4.1 Import Resolution
- Added `src/__init__.py` and `utils/__init__.py` to ensure Python package imports work
- All new modules use absolute imports: `from src.CIFAR_ViT import CIFARViT`, `from utils.pos_embed import VisionRotaryEmbeddingFast`

### 4.2 RoPE `no_rope` Parameter
**Issue:** Original ARC models used `no_rope=num_task_tokens` to exclude task tokens from rotary. CIFAR models optionally have CLS tokens.

**Fix:**
```python
# CIFAR_ViT and CIFAR_LoopViT
no_rope = 1 if use_cls_token else 0
self.encoder = CIFARTransformerEncoder(..., no_rope=no_rope)
```
This ensures CLS token is excluded from RoPE if present, matching ARC pattern.

### 4.3 Positional Embedding Shape
**Issue:** Positional embedding tensor must match total sequence length (CLS + patches if CLS used).

**Fix:**
```python
seq_len = self.num_patches + (1 if self.use_cls_token else 0)
self.positional_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
```
And in `forward()`:
```python
x = self.patch_embed(pixel_values)           # (B, num_patches, D)
if self.use_cls_token:
    x = torch.cat([cls_token, x], dim=1)     # (B, seq_len, D)
x = x + self.positional_embed                 # broadcast add
```

### 4.4 RelativePositionBias API
**Issue in CIFAR_ViT2:** Original `RelativePositionBias.forward(grid_size)` expected single int; changed to `forward(h, w)` for generality.

**Fix in call site:**
```python
# Before (wrong):
bias = self.rel_pos_bias(current_grid_size)
# After:
bias = self.rel_pos_bias(grid_sz, grid_sz)  # square grid, but API now correct
```

---

## Summary of Files Changed/Created

### New Files (10)
| File | Purpose | Lines |
|------|---------|-------|
| `src/CIFAR_ViT.py` | Standard ViT for CIFAR | ~316 |
| `src/CIFAR_ViT2.py` | ViT with RMSNorm+ConvGLU+relative bias | ~406 |
| `src/CIFAR_LoopViT.py` | Looped ViT with exit/step embeddings | ~272 |
| `src/CIFAR_LoopViT_InputInject.py` | Input-injection LoopViT | ~259 |
| `src/CIFAR_loader.py` | CIFAR-10/100 data pipeline | ~388 |
| `src/CIFAR_trainer.py` | Full training script | ~416 |
| `src/CIFAR_evaluator.py` | Evaluation script | ~180 |
| `requirements.txt` | Python dependencies | ~30 |
| `kaggle_notebook.ipynb` | Kaggle-ready notebook | ~297 cells |
| `run_training.sh` | CLI runner | ~56 |

### Modified Files (0)
No existing ARC files were modified. All CIFAR adaptations are new files, preserving original ARC code unchanged.

### Unchanged Reference Files (used as imports)
- `src/ARC_ViT.py` — referenced for `ARCTransformerEncoderLayer` (standard) and `MultiHeadSelfAttention`
- `src/ARC_ViT1.py` — referenced for `RMSNorm`, `ConvolutionalGLU`
- `utils/pos_embed.py` — referenced for `VisionRotaryEmbeddingFast`

---

## Verification Log

All modules tested with `uv` virtual environment (Python 3.12):

```bash
$ .venv/bin/python -c "
import torch
from src.CIFAR_ViT import CIFARViT
from src.CIFAR_ViT2 import CIFARViT2
from src.CIFAR_LoopViT import CIFARLoopViT
from src.CIFAR_LoopViT_InputInject import CIFARLoopViTInputInject

batch = torch.randn(2, 3, 32, 32)
models = {
    'vit': CIFARViT(embed_dim=128, depth=4, num_heads=4, patch_size=4),
    'vit2': CIFARViT2(embed_dim=128, depth=4, num_heads=4, patch_size=4),
    'loopvit': CIFARLoopViT(embed_dim=128, loop_core_depth=2, max_loop_steps=4, num_heads=4),
    'loopvit_input': CIFARLoopViTInputInject(embed_dim=128, loop_core_depth=2, max_loop_steps=4, num_heads=4),
}
for name, m in models.items():
    m.eval()
    out = m(batch)
    logits = out[0] if isinstance(out, tuple) else out
    print(f'{name}: {logits.shape}, params={sum(p.numel() for p in m.parameters()):,}')
"
```

**Results:**
```
vit:         torch.Size([2, 10]), params=1,862,026
vit2:        torch.Size([2, 10]), params=1,907,482
loopvit:     torch.Size([2, 10]), params=939,658
loopvit_input: torch.Size([2, 10]), params=939,658
```

Training step, checkpoint save/load, and import resolution all verified.

---

## Open Issues & Future Work

1. **Data download reliability**: CIFAR download from torchvision may fail (HTTP 503). Plan to pre-download or use local mirror in Kaggle.
2. **Attention mask unused in training**: Current trainer doesn't apply attention masks; kept for API compatibility with ARC's attention masking pattern.
3. **CLS positional embedding ordering**: Initially placed CLS after positional, which is incorrect. Fixed to prepend CLS first, then add full positional tensor.
4. **ViT2 relative bias table size**: Table size fixed at initialization `(2*grid_size-1)^2`. Different grid sizes clamp indices to this range. Works for CIFAR-32/16 but may degrade for very different resolutions.

---

## Migration Guide

To use a CIFAR model instead of an ARC model:

```python
# ARC (dense prediction, per-pixel)
from src.ARC_ViT import ARCViT
model = ARCViT(num_tasks=..., image_size=30, num_colors=10, ...)
logits = model(pixel_values, task_ids)  # (B, C, H, W)

# CIFAR (classification)
from src.CIFAR_ViT import CIFARViT
model = CIFARViT(image_size=32, num_classes=10, ...)
logits = model(pixel_values)  # (B, num_classes)
```

Key API changes:
- `task_ids` argument removed
- `attention_mask` optional but rarely needed (fully observed images)
- Return type always classification logits (no intermediate crops or shapes)

---

**Document version:** 1.0  
**Last updated:** 2026-05-03  
**Implementation status:** ✅ Complete and verified
