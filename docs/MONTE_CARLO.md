# Monte-Carlo training-data generation

GenMC is trained on paired *(optical-property, optical-fluence)* maps in which the
fluence ground truth is produced by Monte-Carlo (MC) light-transport simulation.
This document records the simulation setup used in the paper (Methods, Sec. 5.1,
and Supplementary Sec. 1) so the dataset can be regenerated.

> **Note.** The MATLAB/MCX scripts that produce the raw `.mat` volumes are **not
> included** in this repository; they are available from the corresponding author
> on reasonable request (see *Data availability* in the paper). Once the raw
> `.mat` files are available, convert them to the training format with
> [`genmc/data/prepare_data.py`](../genmc/data/prepare_data.py).

## Simulator

* **Engine:** [MCX](http://mcx.space) (Monte-Carlo eXtreme), GPU-accelerated
  (Fang & Boas, *Opt. Express*, 2009).
* **Simulation volume:** 25.6 (X) × 27.0 (Y) × 11.4 (Z) mm, isotropic voxel size
  0.1 mm.
* **Photons:** 1 × 10⁸ packets per simulation.
* **Hardware / scale:** 5836 volumetric fluence datasets were generated over
  approximately one week on an NVIDIA Quadro RTX 5000.

## Light source (LED-based PA/US probe)

* Two LED bars arranged symmetrically, inter-bar spacing 1.25 mm, tilted 55°
  with respect to the transducer surface.
* Each bar: a 38 × 2 array of LED elements; adjacent-element spacing 0.7 mm.
* Each element modelled as an angular Gaussian beam, zenith variance 0.6 voxels
  (≈120° far-field divergence).

## Numerical tissue phantoms

Layered tissue models were built to reflect realistic anatomy:

* Either two layers (skin, soft tissue) or three layers (coupling medium, skin,
  soft tissue).
* Surface topology variations generated with Gaussian statistics
  (Garcia & Stoll, 1984).
* Skin-layer thickness spatially varying, sampled from a discrete uniform
  distribution U(1, 9) mm; epidermis 0.1–0.3 mm, dermis 1.3–2.9 mm. Skin optical
  properties are the thickness-weighted mean of epidermis and dermis.
* Portions of the anatomy were derived from three public datasets with tissue
  annotations:
  1. High-frequency skin-layer US (atopic dermatitis / psoriasis) —
     <https://data.mendeley.com/datasets/5p7fxjt7vs/1>
  2. 3-label lower-limb muscle US —
     <https://www.cs.cit.tum.de/camp/publications/leg-3d-us-dataset/>
  3. In-vivo LED-based PA/US finger/forearm/wrist/neck (Shi et al.,
     *Photoacoustics*, 2022).

## Optical properties

The required per-voxel properties are the absorption coefficient μₐ, reduced
scattering coefficient μₛ′, anisotropy factor *g*, and refractive index *n*,
sampled from the distributions in Supplementary Table 1. See Supplementary
Table 2 for the discrete values assigned per tissue type and Fitzpatrick skin
type at 690 nm and 850 nm.

## From simulation to training tensors

1. A 2D slice corresponding to the US transducer imaging plane is extracted from
   each volumetric fluence distribution (x–y view).
2. The slice is cropped from the superficial region to 256 × 256.
3. Each fluence map is normalised to its own maximum value.
4. Source (optical-property) and target (fluence) stacks are exported to a
   MATLAB v7.3 `.mat` file and converted to `.npz`:

   ```bash
   python -m genmc.data.prepare_data \
       --mat   data/raw/fluence_dataset.mat \
       --out   data/processed/fluence_dataset.npz \
       --src-key optical_mask_256_filtered \
       --tar-key fcw_raw_norm_log_256_filtered
   ```

   This writes `arr_0` (optical properties, `N×256×256×3`) and `arr_1`
   (optical fluence, `N×256×256×1`), the layout expected by
   [`NPZDataset`](../genmc/data/dataset.py).
