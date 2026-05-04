"""
CIFAR model evaluation script.

Usage:
  python CIFAR_evaluator.py --checkpoint outputs/best/checkpoint_epoch100.pth
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.CIFAR_loader import build_dataloaders, CIFARDataset
from src.CIFAR_ViT import CIFARViT
from src.CIFAR_LoopViT import CIFARLoopViT
from src.CIFAR_LoopViT_InputInject import CIFARLoopViTInputInject
from src.CIFAR_ViT2 import CIFARViT2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CIFAR models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results_path", type=str, default=None, help="Where to save metrics JSON")
    return parser.parse_args()


def load_checkpoint_and_build_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, argparse.Namespace]:
    """Load model from checkpoint, inferring architecture from saved args."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = state["args"]

    # Build model from saved config
    num_classes = 10 if saved_args["dataset"] == "cifar10" else 100
    model_type = saved_args["model"]

    common = dict(
        image_size=saved_args.get("image_size", 32),
        num_classes=num_classes,
        embed_dim=saved_args["embed_dim"],
        depth=saved_args["depth"],
        num_heads=saved_args["num_heads"],
        mlp_dim=int(saved_args["embed_dim"] * saved_args["mlp_ratio"]),
        dropout=saved_args.get("dropout", 0.0),
        patch_size=saved_args["patch_size"],
        use_cls_token=saved_args.get("use_cls_token", True),
        use_mean_pooling=saved_args.get("use_mean_pooling", False),
    )

    if model_type == "vit":
        model = CIFARViT(**common)
    elif model_type == "vit2":
        model = CIFARViT2(**common)
    elif model_type == "loopvit":
        model = CIFARLoopViT(
            image_size=saved_args.get("image_size", 32),
            num_classes=num_classes,
            embed_dim=saved_args["embed_dim"],
            loop_core_depth=saved_args["loop_core_depth"],
            max_loop_steps=saved_args["max_loop_steps"],
            min_loop_steps=saved_args["min_loop_steps"],
            num_heads=saved_args["num_heads"],
            mlp_dim=int(saved_args["embed_dim"] * saved_args["mlp_ratio"]),
            dropout=saved_args.get("dropout", 0.0),
            patch_size=saved_args["patch_size"],
            use_cls_token=saved_args.get("use_cls_token", True),
            use_exit_gate=saved_args.get("use_exit_gate", False),
            gate_threshold=saved_args.get("gate_threshold", 0.5),
            add_step_embeddings=saved_args.get("add_step_embeddings", True),
        )
    elif model_type == "loopvit_input":
        model = CIFARLoopViTInputInject(
            image_size=saved_args.get("image_size", 32),
            num_classes=num_classes,
            embed_dim=saved_args["embed_dim"],
            loop_core_depth=saved_args["loop_core_depth"],
            max_loop_steps=saved_args["max_loop_steps"],
            min_loop_steps=saved_args["min_loop_steps"],
            num_heads=saved_args["num_heads"],
            mlp_dim=int(saved_args["embed_dim"] * saved_args["mlp_ratio"]),
            dropout=saved_args.get("dropout", 0.0),
            patch_size=saved_args["patch_size"],
            use_cls_token=saved_args.get("use_cls_token", True),
            use_exit_gate=saved_args.get("use_exit_gate", False),
            gate_threshold=saved_args.get("gate_threshold", 0.5),
            add_step_embeddings=saved_args.get("add_step_embeddings", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state["model"])
    model = model.to(device)
    model.eval()
    return model, saved_args


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, saved_args = load_checkpoint_and_build_model(args.checkpoint, device)
    print(f"Model type: {saved_args['model']}")

    # If dataset not specified in args, infer and fail
    dataset = args.dataset or saved_args.get("dataset", "cifar10")

    # Loaders
    _, val_loader, _, val_dataset = build_dataloaders(
        data_root=args.data_root,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=saved_args.get("image_size", 32),
        download=True,
        shuffle=False,
    )
    print(f"Validation samples: {len(val_dataset)}")

    # Evaluate
    metrics = evaluate(model, val_loader, device)
    print(f"Val Loss: {metrics['val_loss']:.4f}")
    print(f"Val Accuracy: {metrics['val_acc']:.2f}%")

    # Save results if requested
    if args.results_path:
        Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {args.results_path}")


if __name__ == "__main__":
    main()
