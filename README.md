# GenMC: Real-time generative Monte-Carlo surrogate for quantitative photoacoustic imaging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)

Official code for the paper:

> **GenMC: Real-time generative Monte-Carlo surrogate for quantitative photoacoustic imaging**
> Mengjie Shi, Feng He, Tom Vercauteren, Wenfeng Xia.
> *Nature Machine Intelligence* (2026). *(accepted; volume/DOI to be updated upon publication)*

---

## Overview

Quantitative photoacoustic (PA) imaging of blood oxygen saturation (sO₂) is
limited by the **spectral colouring effect**: wavelength-dependent optical
attenuation distorts the local optical fluence, which cannot be measured in vivo.
Monte-Carlo (MC) simulation is the gold standard for modelling light transport,
but is far too slow for real-time use.

**GenMC** replaces the MC solver with a learned, data-driven mapping from tissue
anatomy (derived from co-registered ultrasound) and literature-based optical
properties to the optical-fluence distribution. It is a **conditional GAN**:

* **Generator** — a U-Net encoder/decoder whose decoder batch-normalisation is
  replaced by **SPADE** (spatially-adaptive normalisation), with **2D coordinate
  layers** on the input to capture the diffusion-driven nature of light
  propagation.
* **Discriminator** — a **PatchGAN** enforcing high-frequency realism on local
  16×16 patches.

The predicted fluence maps compensate multi-wavelength PA signals prior to
linear unmixing, yielding accurate sO₂. GenMC produces a fluence map in
**< 30 ms** (vs. ≈ 5 min for MC), a four-orders-of-magnitude speed-up, and
reaches PSNRs up to **36.24 dB**, outperforming U-Net and Pix2Pix baselines.

```
  Optical-property map (3ch: μa, μs', Γ)              Optical fluence (1ch)
  ┌───────────────────────────┐                      ┌──────────────────┐
  │  + 2D coordinate channels  │   GenMC generator    │  fluence map ϕ   │
  │  ─────────────────────────▶│  U-Net + SPADE  ────▶│  (256 × 256)     │
  └───────────────────────────┘   (cGAN)              └──────────────────┘
                                        ▲
                          PatchGAN discriminator (16×16 patches)
                          real = MC ground truth  /  fake = generated
```

## Repository structure

```
GenMC-optical-fluence-synthesis/
├── genmc/                     # importable package
│   ├── config.py             # command-line configuration / hyperparameters
│   ├── losses.py             # adversarial (BCE) + L1 losses  (Eq. 1)
│   ├── metrics.py            # PSNR, SSIM, MSE                (Eqs. 12–14)
│   ├── utils.py              # visualisation helpers
│   ├── data/
│   │   ├── dataset.py        # NPZDataset (paired property/fluence maps)
│   │   └── prepare_data.py   # MATLAB .mat → .npz conversion
│   └── models/
│       ├── layers.py         # SPADE, CoordConv, U-Net blocks
│       ├── genmc.py          # GenMC generator + PatchGAN discriminator
│       ├── unet.py           # U-Net baseline
│       └── pix2pix.py        # Pix2Pix baseline
├── scripts/
│   ├── train.py              # training entry point
│   └── evaluate.py           # evaluation (PSNR/SSIM/MSE) + visualisation
├── docs/
│   └── MONTE_CARLO.md        # how the MC training data was generated
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/MengjieSHI/GenMC-optical-fluence-synthesis.git
cd GenMC-optical-fluence-synthesis

# (recommended) create an isolated environment
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
# …or install the package itself (editable):
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 1.10. A CUDA-capable GPU is recommended for
training; inference runs comfortably on CPU.

## Data

GenMC is trained on paired *(optical-property, optical-fluence)* maps where the
fluence is produced by MC simulation. The data-generation procedure (simulator,
light source, numerical phantoms, optical properties) is documented in
[`docs/MONTE_CARLO.md`](docs/MONTE_CARLO.md).

The processed dataset format is a compressed `.npz` archive with:

* `arr_0` — optical-property maps, shape `(N, 256, 256, 3)` (absorption μₐ,
  scattering μₛ′, Grüneisen Γ);
* `arr_1` — optical-fluence maps, shape `(N, 256, 256, 1)`.

Convert raw MATLAB `.mat` exports to this format with:

```bash
python -m genmc.data.prepare_data \
    --mat data/raw/fluence_dataset.mat \
    --out data/processed/fluence_dataset.npz \
    --src-key optical_mask_256_filtered \
    --tar-key fcw_raw_norm_log_256_filtered
```

The datasets generated and analysed in the study are available from the
corresponding author on reasonable request.

## Training

```bash
# GenMC (cGAN with SPADE + CoordConv)
python scripts/train.py --model genmc --data-dir data/processed --epochs 10

# baselines
python scripts/train.py --model unet    --data-dir data/processed
python scripts/train.py --model pix2pix --data-dir data/processed
```

The dataset is split 8:1:1 into train/validation/test. Checkpoints are written to
`checkpoints/` (`<model>_best.pt` tracks the best validation PSNR). Run
`python scripts/train.py --help` for the full list of options.

| Hyperparameter      | Value      |
|---------------------|------------|
| Optimiser           | Adam (β₁=0.5, β₂=0.999) |
| Learning rate       | 2 × 10⁻⁴   |
| Batch size          | 1          |
| Epochs              | 10         |
| L1 weight (β)       | 10         |
| Train/val/test split| 8 : 1 : 1  |

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/genmc_best.pt \
    --data-dir   data/processed \
    --num-plot   3
```

Reports mean PSNR, SSIM and MSE against the MC ground truth and (with
`--num-plot`) visualises *(input, prediction, ground truth, difference)*
examples.

| Model    | PSNR (dB) ↑ |
|----------|-------------|
| **GenMC**| **32.89** (up to 36.24) |
| U-Net    | 9.14        |
| Pix2Pix  | 11.80       |

*(Average over forearm/finger/wrist/neck sites; see the paper for SSIM, MSE and
full statistics.)*

## Scope of this repository

**Included:** the GenMC model, the U-Net and Pix2Pix baselines, losses, metrics,
the data loader and `.mat → .npz` converter, and the training/evaluation scripts.

**Not included** (available from the corresponding author on reasonable request):

* the MATLAB/MCX scripts that generate the raw MC fluence volumes (the setup is
  documented in [`docs/MONTE_CARLO.md`](docs/MONTE_CARLO.md));
* the downstream sO₂ pipeline — linear unmixing and fluence compensation
  (Methods, Eqs. 4–11) — and the phantom / in-vivo acquisition and skin-tone
  classification code;
* trained model weights and the study datasets.

**Note on the implementation.** This repository provides a clean PyTorch
reference implementation of the GenMC architecture as described in the Methods.
Some implementation details (e.g. exact channel widths) were chosen to match the
paper's description; minor differences from the original research code may exist.

## Citation

If you use this code, please cite:

```bibtex
@article{shi2026genmc,
  title   = {GenMC: Real-time generative Monte-Carlo surrogate for quantitative photoacoustic imaging},
  author  = {Shi, Mengjie and He, Feng and Vercauteren, Tom and Xia, Wenfeng},
  journal = {xxx},
  year    = {2026},
  note    = {Accepted; volume/DOI to be updated upon publication}
}
```

## Acknowledgements

This work was supported by the Wellcome Trust (203148/Z/16/Z), the Engineering
and Physical Sciences Research Council (EPSRC, NS/A000049/1), and the
King's–China Scholarship Council PhD Scholarship Programme (K-CSC, 202008060071).
The human-volunteer study was approved by the King's College London Research
Ethics Committee (HR-18/19-8881).

The U-Net implementation is adapted from
[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) and the Pix2Pix
baseline from
[junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix);
SPADE and CoordConv follow Park et al. (CVPR 2019) and Liu et al. (NeurIPS 2018).

## License

Released under the [MIT License](LICENSE).

## Contact

Mengjie Shi — `ms1219@ic.ac.uk`
Corresponding author: Wenfeng Xia — `wenfeng.xia@kcl.ac.uk`
