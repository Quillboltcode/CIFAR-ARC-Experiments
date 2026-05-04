# CIFAR Adaptation of ARC ViT/LoopViT Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Building on:** [LoopViT: Scaling Visual ARC with Looped Transformers](https://arxiv.org/abs/2602.02156)

---

## 📝 Overview

This repository contains adaptations of the ARC-focused **LoopViT** and **ARC_ViT** architectures for standard image classification on **CIFAR-10** and **CIFAR-100**.

The original LoopViT models were designed for the Abstraction and Reasoning Corpus (ARC) — a grid-based reasoning benchmark with discrete color predictions and multi-task learning. This work re-engineers those architectures for CIFAR's 32×32 RGB image classification, exploring how ARC's architectural innovations transfer to mainstream computer vision.

**Key questions:**
- Does the looped architecture (feature recycling) benefit CIFAR classification?
- How useful are RMSNorm, ConvolutionalGLU, and relative positional bias for recognition vs reasoning?
- Can ARC's exit gate mechanism adapt to single-task image classification?

---

## 🏗️ Adaptations from ARC to CIFAR

### Architectural Changes

| Component | ARC (Original) | CIFAR (This Repo) |
|-----------|---------------|-------------------|
| **Input** | Discrete color indices (0–9), `(B, H, W)` | RGB pixels (3 channels), `(B, 3, 32, 32)` |
| **Embedding** | `nn.Embedding(num_colors, embed_dim)` | `PatchEmbed(in_chans=3, embed_dim)` |
| **Task Tokens** | Yes — one learnable token per task | No — single-task classification |
| **Context Token** | Task token(s) | Optional CLS token |
| **Output** | Per-pixel color logits `(B, C, H, W)` | Image-level logits `(B, num_classes)` |
| **Pooling** | All patches used | CLS token or mean pooling |
| **RoPE** | Excludes task tokens (`no_rope=1`) | Excludes CLS if used (`no_rope=1`) else `0` |
| **Positional Embed** | Patches only | CLS + patches if CLS enabled |

### Models

- **`CIFAR_ViT`** — Standard Vision Transformer baseline with rotary embeddings (RoPE)
- **`CIFAR_ViT2`** — Enhanced with:
  - RMSNorm (replaces LayerNorm)
  - ConvolutionalGLU (depthwise conv + gated MLP)
  - Relative positional bias (2D bias table) + RoPE hybrid
- **`CIFAR_LoopViT`** — Looped architecture with:
  - Shared transformer layers applied repeatedly
  - Optional step embeddings (positional for loop iterations)
  - Optional exit gate for early exit / dynamic compute
- **`CIFAR_LoopViT_InputInject`** — LoopViT with input injection:
  - Residual addition of initial hidden state at each iteration
  - Improved gradient flow to early layers

All models support configurable `use_cls_token` (default True) and optional `use_mean_pooling`.

---

## 📂 Project Structure

```text
LoopViT/ (CIFAR adaptation)
├── src/
│   ├── CIFAR_ViT.py                  # Standard ViT baseline
│   ├── CIFAR_ViT2.py                 # ViT + RMSNorm + ConvGLU + relative bias
│   ├── CIFAR_LoopViT.py              # Looped ViT (shared layers, exit gate, step embeddings)
│   ├── CIFAR_LoopViT_InputInject.py  # Input-injection LoopViT variant
│   ├── CIFAR_loader.py               # CIFAR-10/100 dataset with augmentations
│   ├── CIFAR_trainer.py              # Training script (AdamW, warmup+cosine, AMP, checkpointing)
│   ├── CIFAR_evaluator.py            # Evaluation script for test set
│   ├── ARC_ViT.py                    # Original ARC ViT (unchanged reference)
│   ├── ARC_ViT1.py                   # Original RMSNorm + ConvGLU (unchanged reference)
│   ├── ARC_LoopViT.py                # Original LoopViT (unchanged reference)
│   └── ARC_loader.py                 # Original ARC loader (unchanged reference)
├── utils/
│   └── pos_embed.py                  # VisionRotaryEmbeddingFast (shared)
├── .kilo/plans/
│   ├── 1777708273669-nimble-otter.md     # Original adaptation plan
│   └── CIFAR_ADAPTATION_CHANGELOG.md     # Detailed diff from ARC → CIFAR
├── requirements.txt                  # Python dependencies (torch, timm, einops, etc.)
├── run_training.sh                   # CLI launcher for training
├── kaggle_notebook.ipynb             # Ready-to-run Kaggle notebook
└── README.md                         # This file
```

**Note:** The original ARC models (`ARC_ViT.py`, `ARC_LoopViT.py`, etc.) are preserved unmodified for reference and backward compatibility.

---

## 🚀 Quick Start

### Local Setup (Python 3.12 with uv)

```bash
# 1. Clone this repository
git clone https://github.com/Quillboltcode/CIFAR-ARC-Experiments.git
cd LoopViT

# 2. Create virtual environment with uv
uv venv .venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Download CIFAR (first run)
python -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True)"

# 5. Train a model
./run_training.sh --model vit --dataset cifar10 --epochs 100 --embed_dim 384 --depth 8

# 6. Evaluate
python -m src.CIFAR_evaluator --checkpoint outputs/best/checkpoint_epoch100.pth
```

### Training with Different Architectures

```bash
# Standard ViT
./run_training.sh --model vit --dataset cifar10 --embed_dim 384 --depth 8

# Enhanced ViT2 with relative bias
./run_training.sh --model vit2 --dataset cifar10 --embed_dim 384 --depth 8

# Looped ViT (shared layers)
./run_training.sh --model loopvit --dataset cifar10 --embed_dim 384 --loop_core_depth 2 --max_loop_steps 6

# LoopViT with input injection
./run_training.sh --model loopvit_input --dataset cifar10 --embed_dim 384 --loop_core_depth 2
```

### On Kaggle

1. Upload this repository as a Kaggle dataset or use the notebook directly
2. Open `kaggle_notebook.ipynb` in Kaggle Notebooks
3. Configure hyperparameters in the first cell
4. Run all cells — training will execute with GPU acceleration

---

## 📊 Expected Performance (Indicative)

Based on standard CIFAR-10 results for ViT-like models with patch size 4:

| Model | Params (≈) | Expected Top-1 Acc |
|-------|------------|-------------------|
| `CIFAR_ViT` (Ti) | 1.9M | 92–94% |
| `CIFAR_ViT2` (Ti) | 1.9M | 92–95% (potentially better) |
| `CIFAR_LoopViT` (loop depth=2, steps=4) | 0.9M | 90–93% (more efficient) |
| `CIFAR_LoopViT_InputInject` | 0.9M | Similar or slightly better than LoopViT |

*These are rough estimates; actual results depend on training schedule, augmentation, and regularization.*

---

## 🧪 Experiments to Run

### Ablation Studies
- [ ] Compare CLS token vs mean pooling
- [ ] Effect of `loop_core_depth` (1, 2, 4) and `max_loop_steps` (2, 4, 6, 8)
- [ ] Exit gate utility (train with `--use_exit_gate` and measure accuracy vs average steps)
- [ ] Step embeddings: `--add_step_embeddings True/False`
- [ ] ViT2 components: RMSNorm vs LayerNorm, ConvGLU vs MLP, relative bias on/off

### Scaling
- [ ] Model sizes: Tiny (embed_dim=192), Small (384), Base (512)
- [ ] Patch sizes: 2, 4, 8 (affects sequence length)
- [ ] Dataset: CIFAR-10 (10 classes) vs CIFAR-100 (100 classes)

### Training Recipes
- [ ] Longer training (300–400 epochs)
- [ ] Label smoothing, mixup, CutMix
- [ ] Different LR schedules (step decay, cosine without warmup)
- [ ] Weight decay variations

---

## 📖 Citation & Attribution

This work builds directly on the following paper:

**LoopViT: Scaling Visual ARC with Looped Transformers**
- Wen-Jie Shu et al.
- arXiv:2602.02156 (2026)

If you use this code or ideas, please cite both the original LoopViT paper and this adaptation:

```bibtex
@article{shu2026loopvit,
  title={LoopViT: Scaling Visual ARC with Looped Transformers},
  author={Shu, Wen-Jie and Qiu, Xuerui and Zhu, Rui-Jie and Chen, Harold Haodong and Liu, Yexin and Yang, Harry},
  journal={arXiv preprint arXiv:2602.02156},
  year={2026}
}

@misc{cifar-arc-adaptation,
  title={CIFAR Adaptation of ARC ViT/LoopViT Models},
  author={Quillbolt},
  year={2026},
  publisher={GitHub},
  url={https://github.com/Quillboltcode/CIFAR-ARC-Experiments}
}
```

The original ARC dataset and challenge: [Chollet, F. (2019)](https://github.com/fchollet/ARC).

---

## 🔧 Implementation Notes

### Detailed documentation of changes from ARC → CIFAR
See `.kilo/plans/CIFAR_ADAPTATION_CHANGELOG.md` for a comprehensive technical changelog:
- Positional embedding ordering fixes
- RoPE `no_rope` handling for CLS tokens
- RelativePositionBias API changes
- Output head redesign (dense → global)
- Task token removal and its ripple effects

### Design Decisions

**Why keep LoopViT for CIFAR?**
ARC is a reasoning task requiring iterative refinement; CIFAR is typically single-pass recognition. The looped architecture may:
- Improve gradient flow via repeated refinement
- Allow early exit on easy examples (if gate trained)
- Provide computational flexibility (variable depth per sample)

**Why ViT2 innovations?**
RMSNorm and ConvGLU add inductive bias (locality, gating) that may help small-data regimes like CIFAR despite being designed for reasoning.

---

## 📫 Contact

For questions about this adaptation, open an issue on the repository.

---

## License

MIT License — same as original LoopViT codebase.

---

**Status:** ✅ All models implemented, tested, and verified with synthetic data.  
**Ready for:** Local training runs, Kaggle experiments, ablation studies.
