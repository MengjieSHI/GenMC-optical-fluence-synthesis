"""Evaluate a trained model against Monte-Carlo ground-truth fluence maps.

Computes PSNR, SSIM and MSE (Methods, Eqs. 12-14) over a dataset and, optionally,
visualises a few (input, prediction, ground-truth, difference) examples.

Example
-------
::

    python scripts/evaluate.py \\
        --checkpoint checkpoints/genmc_best.pt \\
        --data-dir data/processed \\
        --num-plot 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genmc.data import NPZDataset
from genmc.metrics import compute_all
from genmc.models import build_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GenMC / baseline model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint saved by train.py.")
    parser.add_argument("--data-dir", required=True, help="Directory (or .npz file) with test data.")
    parser.add_argument("--model", default=None, choices=[None, "genmc", "pix2pix", "unet"],
                        help="Override the model type (otherwise read from the checkpoint).")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--num-plot", type=int, default=0, help="Number of examples to visualise.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def plot_examples(generator, dataset, n: int, device) -> None:
    import matplotlib.pyplot as plt

    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    titles = ["Source (mu_a)", "Generated", "Ground truth (MC)", "Difference"]
    for row in range(n):
        inp, target = dataset[row]
        with torch.no_grad():
            pred = generator(inp.unsqueeze(0).to(device)).cpu()[0]
        src = inp[0].numpy()
        gen = pred[0].numpy()
        gt = target[0].numpy()
        for col, image in enumerate([src, gen, gt, gen - gt]):
            ax = axes[row][col]
            ax.imshow(np.rot90(image, k=-1))
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col])
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or checkpoint.get("model", "genmc")
    generator = build_generator(model_name, args.in_channels, args.out_channels).to(device)
    generator.load_state_dict(checkpoint["state_dict"])
    generator.eval()
    print(f"Loaded {model_name} from {args.checkpoint}")

    dataset = NPZDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    psnr_vals, ssim_vals, mse_vals = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            preds = generator(inputs.to(device)).cpu()
            for pred, target in zip(preds, targets):
                m = compute_all(target, pred)
                psnr_vals.append(m["psnr"])
                mse_vals.append(m["mse"])
                if "ssim" in m:
                    ssim_vals.append(m["ssim"])

    print(f"Samples evaluated: {len(psnr_vals)}")
    print(f"PSNR: {np.mean(psnr_vals):.2f} dB")
    if ssim_vals:
        print(f"SSIM: {np.mean(ssim_vals):.4f}")
    print(f"MSE : {np.mean(mse_vals):.6f}")

    if args.num_plot > 0:
        plot_examples(generator, dataset, args.num_plot, device)


if __name__ == "__main__":
    main()
