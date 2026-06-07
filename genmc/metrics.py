"""Image-quality metrics for evaluating predicted optical-fluence maps.

Predicted maps are compared against the Monte-Carlo ground truth using the peak
signal-to-noise ratio (PSNR), structural similarity index measure (SSIM) and mean
squared error (MSE), as defined in the Methods (Eqs. 12-14).

Functions accept either 2D NumPy arrays or PyTorch tensors (which are detached
and moved to CPU automatically).
"""

from __future__ import annotations

import numpy as np

try:  # SSIM is delegated to scikit-image, the de-facto reference implementation.
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.squeeze(np.asarray(x, dtype=np.float64))


def mse(reference, prediction) -> float:
    """Mean squared error between a reference and a predicted map."""
    ref, pred = _to_numpy(reference), _to_numpy(prediction)
    return float(np.mean((ref - pred) ** 2))


def psnr(reference, prediction, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio (dB).

    ``data_range`` defaults to the dynamic range of the reference map.
    """
    ref, pred = _to_numpy(reference), _to_numpy(prediction)
    if data_range is None:
        data_range = float(ref.max() - ref.min())
    error = np.mean((ref - pred) ** 2)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / error))


def ssim(reference, prediction, data_range: float | None = None) -> float:
    """Structural similarity index measure."""
    if not _HAS_SKIMAGE:
        raise ImportError("scikit-image is required for SSIM. Install it with `pip install scikit-image`.")
    ref, pred = _to_numpy(reference), _to_numpy(prediction)
    if data_range is None:
        data_range = float(ref.max() - ref.min())
    return float(_ssim(ref, pred, data_range=data_range))


def compute_all(reference, prediction) -> dict[str, float]:
    """Return PSNR, SSIM and MSE for a single (reference, prediction) pair."""
    metrics = {"psnr": psnr(reference, prediction), "mse": mse(reference, prediction)}
    if _HAS_SKIMAGE:
        metrics["ssim"] = ssim(reference, prediction)
    return metrics
