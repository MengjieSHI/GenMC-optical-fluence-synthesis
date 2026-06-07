"""Train GenMC (or the U-Net / Pix2Pix baselines) for optical-fluence synthesis.

Examples
--------
Train GenMC::

    python scripts/train.py --model genmc --data-dir data/processed --epochs 10

Train a baseline::

    python scripts/train.py --model unet    --data-dir data/processed
    python scripts/train.py --model pix2pix --data-dir data/processed

The dataset is split 8:1:1 into train/validation/test (Methods, Sec. 5.2).
Checkpoints are written to ``--checkpoint-dir``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Allow running as `python scripts/train.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genmc.config import get_options
from genmc.data import NPZDataset
from genmc.losses import DiscriminatorLoss, GeneratorLoss
from genmc.metrics import psnr
from genmc.models import build_discriminator, build_generator


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_lr_lambda(epochs: int, decay_start: int):
    """Constant learning rate, then linear decay to zero after ``decay_start``."""
    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start or epochs <= decay_start:
            return 1.0
        return max(0.0, 1.0 - float(epoch - decay_start) / float(epochs - decay_start))
    return lr_lambda


def split_dataset(dataset, val_split: float, test_split: float, seed: int):
    n_total = len(dataset)
    n_val = int(round(n_total * val_split))
    n_test = int(round(n_total * test_split))
    n_train = n_total - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


@torch.no_grad()
def evaluate(generator, loader, device) -> float:
    """Return the mean PSNR of the generator over a data loader."""
    generator.eval()
    scores, count = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = generator(inputs)
        for pred, target in zip(preds, targets):
            scores += psnr(target, pred)
            count += 1
    return scores / max(count, 1)


def train_gan(args, train_loader, val_loader, device):
    """Adversarial training for GenMC / Pix2Pix."""
    generator = build_generator(args.model, args.in_channels, args.out_channels).to(device)
    discriminator = build_discriminator(args.model, args.out_channels, args.in_channels).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    lr_lambda = make_lr_lambda(args.epochs, args.lr_decay_start)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda)

    g_criterion = GeneratorLoss(l1_weight=args.l1_weight)
    d_criterion = DiscriminatorLoss()

    best_psnr = float("-inf")
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        start = time.time()
        g_running, d_running = 0.0, 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)

            # --- Update discriminator ---
            fake = generator(inputs)
            fake_pred = discriminator(fake.detach(), inputs)
            real_pred = discriminator(targets, inputs)
            d_loss = d_criterion(fake_pred, real_pred)
            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()

            # --- Update generator ---
            fake_pred = discriminator(fake, inputs)
            g_loss = g_criterion(fake, targets, fake_pred)
            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()

            g_running += g_loss.item()
            d_running += d_loss.item()
            progress.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

        g_scheduler.step()
        d_scheduler.step()

        n_batches = max(len(train_loader), 1)
        val_psnr = evaluate(generator, val_loader, device) if len(val_loader) else float("nan")
        print(f"[Epoch {epoch + 1}/{args.epochs}] "
              f"G: {g_running / n_batches:.3f}  D: {d_running / n_batches:.3f}  "
              f"val PSNR: {val_psnr:.2f} dB  ({time.time() - start:.1f}s)")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(generator, args, f"{args.model}_epoch{epoch + 1}.pt")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(generator, args, f"{args.model}_best.pt")

    print(f"Training complete. Best validation PSNR: {best_psnr:.2f} dB")


def train_unet(args, train_loader, val_loader, device):
    """Supervised (non-adversarial) training for the U-Net baseline."""
    generator = build_generator("unet", args.in_channels, args.out_channels).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(args.epochs, args.lr_decay_start))
    criterion = nn.MSELoss()

    best_psnr = float("-inf")
    for epoch in range(args.epochs):
        generator.train()
        start = time.time()
        running = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = generator(inputs)
            loss = criterion(preds, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        val_psnr = evaluate(generator, val_loader, device) if len(val_loader) else float("nan")
        print(f"[Epoch {epoch + 1}/{args.epochs}] loss: {running / max(len(train_loader), 1):.4f}  "
              f"val PSNR: {val_psnr:.2f} dB  ({time.time() - start:.1f}s)")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(generator, args, f"unet_epoch{epoch + 1}.pt")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(generator, args, "unet_best.pt")

    print(f"Training complete. Best validation PSNR: {best_psnr:.2f} dB")


def save_checkpoint(generator, args, filename: str) -> None:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "state_dict": generator.state_dict()}, ckpt_dir / filename)


def main() -> None:
    args = get_options()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Model: {args.model} | Device: {device}")

    dataset = NPZDataset(args.data_dir)
    train_set, val_set, test_set = split_dataset(dataset, args.val_split, args.test_split, args.seed)
    print(f"Dataset: {len(dataset)} samples -> "
          f"train {len(train_set)}, val {len(val_set)}, test {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    if args.model == "unet":
        train_unet(args, train_loader, val_loader, device)
    else:
        train_gan(args, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
