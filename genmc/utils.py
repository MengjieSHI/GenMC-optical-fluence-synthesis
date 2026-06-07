"""Visualisation and miscellaneous helpers."""

from __future__ import annotations

import numpy as np
import torch


def show_tensor_images(image_tensor: torch.Tensor, cmap: str = "viridis", rot90: int = 3):
    """Display the first image of a ``(B, C, H, W)`` batch with matplotlib."""
    import matplotlib.pyplot as plt

    image = image_tensor.detach().cpu()[0, 0]
    if rot90:
        image = torch.rot90(image, rot90)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()


def plot_property_and_fluence(src_images: np.ndarray, tar_images: np.ndarray, n_samples: int = 3):
    """Plot optical-property channels and the corresponding fluence maps.

    Args:
        src_images: ``(N, H, W, 3)`` array (absorption, scattering, Grueneisen).
        tar_images: ``(N, H, W, 1)`` optical-fluence array.
        n_samples: number of samples (columns) to display.
    """
    import matplotlib.pyplot as plt

    titles = ["absorption", "scattering", "Grueneisen", "optical fluence"]
    n_rows = 4
    plt.figure(figsize=(2.5 * n_samples, 2.5 * n_rows))
    for i in range(n_samples):
        for ch in range(3):
            ax = plt.subplot(n_rows, n_samples, ch * n_samples + 1 + i)
            ax.axis("off")
            ax.imshow(np.rot90(src_images[i][:, :, ch], k=-1))
            if i == 0:
                ax.set_ylabel(titles[ch])
            ax.set_title(titles[ch] if i == n_samples // 2 else "")
        ax = plt.subplot(n_rows, n_samples, 3 * n_samples + 1 + i)
        ax.axis("off")
        ax.imshow(np.rot90(np.squeeze(tar_images[i]), k=-1))
        ax.set_title(titles[3] if i == n_samples // 2 else "")
    plt.tight_layout()
    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
