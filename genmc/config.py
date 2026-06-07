"""Command-line configuration for training and evaluation.

Defaults reflect the experimental setup reported in the Methods (Sec. 5.2):
10 training epochs, batch size 1, Adam with a learning rate of 2e-4, and an L1
loss weight (beta) of 10. An 8:1:1 train/validation/test split is used.
"""

from __future__ import annotations

import argparse


def get_options(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenMC optical-fluence synthesis.")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory (or .npz file) with paired property/fluence data.")
    parser.add_argument("--in-channels", type=int, default=3, help="Optical-property input channels.")
    parser.add_argument("--out-channels", type=int, default=1, help="Optical-fluence output channels.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test fraction.")

    # Model
    parser.add_argument("--model", type=str, default="genmc", choices=["genmc", "pix2pix", "unet"],
                        help="Model to train.")

    # Optimisation
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--l1-weight", type=float, default=10.0, help="L1 loss weight (beta).")
    parser.add_argument("--lr-decay-start", type=int, default=100,
                        help="Epoch after which the learning rate decays linearly to 0.")

    # I/O & runtime
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (default: cuda if available, else cpu).")
    parser.add_argument("--save-every", type=int, default=1, help="Checkpoint every N epochs.")

    args = parser.parse_args(argv)
    return args
