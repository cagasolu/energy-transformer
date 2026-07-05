# energy-transformer

Pretrained weights (~750 MB): https://huggingface.co/cagasoluh/energy-transformer

---

## SELYNE

Selyne (Stable-Energy Lipschitz Net) introduces Gloeba, an energy-based attention mechanism that eliminates MCMC sampling and thermal quenching. Official PyTorch implementation.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20779017.svg)](https://doi.org/10.5281/zenodo.20779017)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/license-GPLv3-blue)

---

## Key Highlights

- Removes MCMC sampling from EBMs; closed-form energy minimization
- Gloeba: learnable bilinear compatibility M_h + adaptive temperature tau_h
- Lipschitz-bounded attention (Proposition 4)
- Expands representable forms beyond PSD cone (Proposition 5)
- 59.3M params, 4.4% fewer than untied standard, higher accuracy
- Mahalanobis AUROC: 0.895 on STL-10 (90% anomaly rate)
- Google Cloud / Colab ready with auto Drive mounting

---

## Architecture

1. Patch Embedding
2. Gloeba Encoder (6 blocks)
3. Prototype Cross-Attention
4. Latent MLP
5. Decoder

Gloeba: S_h = Q_h M_h K_h^T, tau_h = e^(ell_h), clamped >= 1e-4

Scoring: Reconstruction (MSE+TV+FFT+pool) + Mahalanobis (Ledoit-Wolf shrinkage)

---

## Ablation: Tied Gloeba vs. Untied Standard

| Component | Untied | Tied Gloeba |
|-----------|--------|-------------|
| Q/K | W_Q, W_K | W_Q = W_K = W |
| Kernel | W_Q W_K^T | W M_h W^T |
| Params | 62.1M | 59.3M |

Pretraining (7 seeds, Tiny ImageNet):

| Metric | Tied | Untied | Delta |
|--------|------|--------|-------|
| Val acc (mean) | 0.5650 | 0.5594 | +0.0056 (p=0.011) |
| Val acc (max) | 0.572 | 0.563 | +0.0090 |
| Training time | 3.3h | 2.95h | +0.35h |

STL-10 (7 seeds):

| Metric | Tied | Untied | Delta |
|--------|------|--------|-------|
| Mahalanobis AUROC | 0.8948 | 0.8974 | +0.0027 |
| Reconstruction AUROC | 0.5270 | 0.5286 | +0.0016 |

Gap localized to Bird (favors untied) and Ship (favors tied). Removing both: -0.0002, p=0.71 -> practically equivalent detectors.

---

## Quick Start

Colab:

    !git clone https://github.com/cagasolu/energy-transformer.git
    %cd energy-transformer
    !pip install -r requirements.txt
    from google.colab import drive
    drive.mount('/content/drive')
    !python pretrain_selyne_recon.py

Local:

    git clone https://github.com/cagasolu/energy-transformer.git
    cd energy-transformer
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

---

## Usage

Pretrain:

    python pretrain_selyne_recon.py          # Tied Gloeba
    python pretrain_standard_recon.py        # Untied standard

STL-10 Anomaly Detection:

    python globaleba_mahalanobis.py          # Tied Gloeba
    python standard_mahal_science.py         # Untied standard

BRISC2025 Brain MRI:

    python energetic_mahal_science.py

---

## Results

STL-10 (90% anomaly):

    Score               | Mean AUROC | CV
    Reconstruction      | 0.5270     | 0.34%
    Mahalanobis         | 0.8948     | 0.14%

Per-class Mahalanobis (seed 2584):

    Airplane 0.936, Bird 0.784, Car 0.961, Cat 0.848, Deer 0.902, Dog 0.871, Horse 0.912, Monkey 0.880, Ship 0.944, Truck 0.929

Mean: 0.897

BRISC2025 (86% anomaly):

    Score               | Mean AUROC | CV
    Reconstruction      | 0.753      | 2.56%
    Mahalanobis         | 0.852      | 1.71%

Compute: ~43.75 GPU-hours (14 runs, A100)

---

## Citation

    @misc{suleymanoglu2026selyne,
      author = {Suleymanoglu, Gorkem Can},
      title = {Selyne: Stable-Energy Lipschitz Network with Energy-Based Attention for Anomaly Detection},
      year = {2026},
      publisher = {GitHub},
      url = {https://github.com/cagasolu/energy-transformer},
      doi = {10.5281/zenodo.20779017}
    }

---

## Links

GitHub: https://github.com/cagasolu/energy-transformer
Hugging Face: https://huggingface.co/cagasoluh/energy-transformer
Zenodo: https://doi.org/10.5281/zenodo.20779017

---

## License

GPLv3
