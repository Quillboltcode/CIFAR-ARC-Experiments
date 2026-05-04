"""
CIFAR-10/CIFAR-100 training and evaluation script.

Usage:
  python CIFAR_trainer.py --dataset cifar10 --model vit --epochs 100
  python CIFAR_trainer.py --dataset cifar100 --model loopvit --epochs 200
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.CIFAR_loader import build_dataloaders, CIFARDataset
from src.CIFAR_ViT import CIFARViT
from src.CIFAR_LoopViT import CIFARLoopViT
from src.CIFAR_LoopViT_InputInject import CIFARLoopViTInputInject
from src.CIFAR_ViT2 import CIFARViT2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT/LoopViT on CIFAR")

    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=32)

    # Model
    parser.add_argument("--model", type=str, default="vit", 
                        choices=["vit", "vit2", "loopvit", "loopvit_input"])
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_cls_token", action="store_true", default=True)
    parser.add_argument("--use_mean_pooling", action="store_true", default=False)

    # LoopViT specific
    parser.add_argument("--loop_core_depth", type=int, default=2)
    parser.add_argument("--max_loop_steps", type=int, default=6)
    parser.add_argument("--min_loop_steps", type=int, default=1)
    parser.add_argument("--use_exit_gate", action="store_true", default=False)
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--add_step_embeddings", action="store_true", default=True)

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Automatic mixed precision")

    # Execution
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")

    # Distributed (not fully implemented, but placeholder)
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> nn.Module:
    """Construct model based on args."""
    num_classes = 10 if args.dataset == "cifar10" else 100

    common = dict(
        image_size=args.image_size,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=int(args.embed_dim * args.mlp_ratio),
        dropout=args.dropout,
        patch_size=args.patch_size,
    )

    if args.model == "vit":
        model = CIFARViT(
            **common,
            use_cls_token=args.use_cls_token,
            use_mean_pooling=args.use_mean_pooling,
        )
    elif args.model == "vit2":
        model = CIFARViT2(
            **common,
            use_cls_token=args.use_cls_token,
        )
    elif args.model == "loopvit":
        model = CIFARLoopViT(
            image_size=args.image_size,
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            loop_core_depth=args.loop_core_depth,
            max_loop_steps=args.max_loop_steps,
            min_loop_steps=args.min_loop_steps,
            num_heads=args.num_heads,
            mlp_dim=int(args.embed_dim * args.mlp_ratio),
            dropout=args.dropout,
            patch_size=args.patch_size,
            use_cls_token=args.use_cls_token,
            use_exit_gate=args.use_exit_gate,
            gate_threshold=args.gate_threshold,
            add_step_embeddings=args.add_step_embeddings,
        )
    elif args.model == "loopvit_input":
        model = CIFARLoopViTInputInject(
            image_size=args.image_size,
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            loop_core_depth=args.loop_core_depth,
            max_loop_steps=args.max_loop_steps,
            min_loop_steps=args.min_loop_steps,
            num_heads=args.num_heads,
            mlp_dim=int(args.embed_dim * args.mlp_ratio),
            dropout=args.dropout,
            patch_size=args.patch_size,
            use_cls_token=args.use_cls_token,
            use_exit_gate=args.use_exit_gate,
            gate_threshold=args.gate_threshold,
            add_step_embeddings=args.add_step_embeddings,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model


def setup_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Setup optimizer with cosine decay + warmup."""
    # Separate weight decay params
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "cls_token" in name or "pos_embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    
    # Scheduler: warmup + cosine
    warmup_steps = args.warmup_epochs * total_steps // args.epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        pixel_values = batch["pixel_values"].to(device)  # (B, 3, H, W)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(pixel_values, attention_mask=None)
                logits = output if isinstance(output, torch.Tensor) else output[0]
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(pixel_values)
            logits = output if isinstance(output, torch.Tensor) else output[0]
            loss = criterion(logits, labels)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {total_loss / (batch_idx+1):.4f} | "
                  f"Acc: {100 * correct / total:.2f}%")

    epoch_time = time.time() - start_time
    metrics = {
        "train_loss": total_loss / len(train_loader),
        "train_acc": 100.0 * correct / total,
        "epoch_time": epoch_time,
        "lr": scheduler.get_last_lr()[0],
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        output = model(pixel_values)
        logits = output if isinstance(output, torch.Tensor) else output[0]
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "val_loss": total_loss / len(val_loader),
        "val_acc": 100.0 * correct / total,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> None:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pth"
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "args": vars(args),
    }
    torch.save(state, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """Load checkpoint and return epoch."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    print(f"Loaded checkpoint from epoch {state['epoch']} (acc: {state['best_acc']:.2f}%)")
    return state["epoch"]


def main() -> None:
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    print("Loading data...")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
        data_root=args.data_root,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        download=True,
        shuffle=True,
        drop_last=False,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print(f"Building {args.model} model...")
    model = build_model(args)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")

    # Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer & Scheduler
    total_steps = len(train_loader) * args.epochs
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args, total_steps)

    # Resume
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler) + 1

    # Evaluate only
    if args.eval_only:
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.2f}%")
        return

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args)
        val_metrics = evaluate(model, val_loader, criterion, device)

        metrics = {**train_metrics, **val_metrics}
        metrics["epoch"] = epoch
        history.append(metrics)

        print(f"Epoch {epoch} | Train Loss: {train_metrics['train_loss']:.4f}, "
              f"Train Acc: {train_metrics['train_acc']:.2f}% | "
              f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Val Acc: {val_metrics['val_acc']:.2f}% | "
              f"LR: {metrics['lr']:.6f}")

        # Save best
        if val_metrics["val_acc"] > best_acc:
            best_acc = val_metrics["val_acc"]
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, args, output_dir / "best")

        # Periodic save
        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, args, output_dir / f"checkpoint_epoch{epoch}")

    # Save final
    save_checkpoint(model, optimizer, scheduler, args.epochs, best_acc, args, output_dir / "final")
    
    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete! Best val accuracy: {best_acc:.2f}%")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
