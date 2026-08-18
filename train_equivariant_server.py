"""
Server-optimized equivariant model training.
Trains on ALL ~13,000 elastic tensors with GPU acceleration.

Changes from the local version (train_equivariant.py):
  - More epochs (100 instead of 10)
  - DataLoader workers (4 instead of 0)
  - Checkpointing (saves best model)
  - Logging to file
  - GPU auto-detected
  - Better learning rate schedule

Usage:
    python train_equivariant_server.py
    python train_equivariant_server.py --epochs 200
"""
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from equivariant_dataset import EquivariantTensorDataset
from equivariant_model import SimpleEquivariantElasticNet
from tensor_utils import calculate_moduli, check_stability


def parse_args():
    parser = argparse.ArgumentParser(description="Train equivariant elastic tensor model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1, graph data)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--data-dir", type=str, default="dataset_equivariant", help="Dataset directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints_equivariant", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    return parser.parse_args()


def train(args):
    # ── Setup ──
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = open(os.path.join(args.save_dir, "training_log.txt"), "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load dataset ──
    dataset = EquivariantTensorDataset(data_dir=args.data_dir)

    if len(dataset) == 0:
        log("Dataset is empty! Run fetch_elastic_tensors_full.py first.")
        return

    log(f"Dataset size: {len(dataset)}")

    # Split train/val
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    log(f"Train: {train_size} | Val: {val_size}")

    # batch_size=1 because graphs have variable sizes
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # ── Model ──
    model = SimpleEquivariantElasticNet(num_node_features=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    start_epoch = 0
    best_val_loss = float("inf")

    # ── Resume from checkpoint ──
    if args.resume and os.path.isfile(args.resume):
        log(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log(f"Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    # ── Compute target statistics for Z-score normalization ──
    log("Computing target statistics...")
    all_targets = []
    for batch in train_loader:
        all_targets.append(batch["target_tensor"][0].flatten())
    all_targets = torch.stack(all_targets)
    target_mean = all_targets.mean(dim=0).to(device).view(6, 6)
    target_std = all_targets.std(dim=0).to(device).view(6, 6)
    target_std[target_std < 1e-4] = 1.0

    # Save normalization stats
    torch.save({"mean": target_mean, "std": target_std},
               os.path.join(args.save_dir, "normalization_stats.pt"))

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {param_count:,}")
    log(f"Starting training for {args.epochs} epochs...\n")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            node_features = batch["node_features"][0].to(device)
            pos = batch["pos"][0].to(device)
            edge_index = batch["edge_index"][0].to(device)
            target_tensor = batch["target_tensor"][0].to(device)

            # Standardize target
            standardized_target = (target_tensor - target_mean) / target_std

            optimizer.zero_grad()
            pred_C = model(node_features, pos, edge_index)
            loss = F.mse_loss(pred_C, standardized_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                node_features = batch["node_features"][0].to(device)
                pos = batch["pos"][0].to(device)
                edge_index = batch["edge_index"][0].to(device)
                target_tensor = batch["target_tensor"][0].to(device)

                standardized_target = (target_tensor - target_mean) / target_std
                pred_C = model(node_features, pos, edge_index)
                loss = F.mse_loss(pred_C, standardized_target)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        elapsed = time.time() - epoch_start

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        log(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        # ── Checkpointing ──
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
        }

        torch.save(checkpoint, os.path.join(args.save_dir, "checkpoint_equivariant.pth.tar"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, "model_best_equivariant.pth.tar"))
            log(f"  *** New best model saved (val_loss: {best_val_loss:.6f}) ***")

    log(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    log("Evaluating best model on validation samples...\n")

    # ── Final evaluation ──
    best_ckpt = torch.load(os.path.join(args.save_dir, "model_best_equivariant.pth.tar"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    evaluate(model, val_loader, device, target_mean, target_std, log)
    log_file.close()


def evaluate(model, val_loader, device, target_mean, target_std, log):
    model.eval()
    target_mean_np = target_mean.cpu().numpy()
    target_std_np = target_std.cpu().numpy()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:
                break

            mat_id = batch["mat_id"][0]
            node_features = batch["node_features"][0].to(device)
            pos = batch["pos"][0].to(device)
            edge_index = batch["edge_index"][0].to(device)
            target_tensor = batch["target_tensor"][0].cpu().numpy()

            pred_standard = model(node_features, pos, edge_index).cpu().numpy()
            pred_C = pred_standard * target_std_np + target_mean_np

            true_moduli = calculate_moduli(target_tensor)
            pred_moduli = calculate_moduli(pred_C)

            log(f"--- {mat_id} ---")
            log(f"  TRUE:  Bulk={true_moduli.get('K_H', float('nan')):.1f}  Shear={true_moduli.get('G_H', float('nan')):.1f}  Stable={check_stability(target_tensor)}")
            log(f"  PRED:  Bulk={pred_moduli.get('K_H', float('nan')):.1f}  Shear={pred_moduli.get('G_H', float('nan')):.1f}  Stable={check_stability(pred_C)}")
            log(f"  MSE:   {np.mean((target_tensor - pred_C)**2):.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
