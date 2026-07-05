# energy-transformer

The pretrained model weights are available on Hugging Face due to their large size (∼1 GB).

Download them from the repository below:

**Hugging Face**: https://huggingface.co/cagasoluh/energy-transformer

---

## SELYNE: Stable-Energy Lipschitz Network with Energy-Based Attention for Anomaly Detection

Selyne (Stable-Energy Lipschitz Net) introduces a novel energy-based attention mechanism, **Gloeba** (Global Energy-Based Attention), designed to overcome the thermal-quenching instability observed in traditional energy-based models. This repository contains the official PyTorch implementation, pretrained models, and evaluation code.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20779017.svg)](https://doi.org/10.5281/zenodo.20779017)

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-GPLv3-blue)

---

## Table of Contents
- [Key Highlights](#key-highlights)
- [Overview](#overview)
- [Core Concepts & Architecture](#core-concepts--architecture)
- [Ablation: Tied Gloeba vs. Untied Standard](#ablation-tied-gloeba-vs-untied-standard)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Key Highlights
- **Addresses Thermal Quenching**: Selyne removes the unstable MCMC sampling step inherent in classical Energy-Based Models, replacing it with a deterministic, closed-form energy minimization process.
- **Novel Attention Mechanism**: Introduces Gloeba (Global Energy-Based Attention), which uses a learnable, per-head bilinear compatibility matrix ($M_h$) and an adaptive, per-head temperature ($\tau_h$) as an out-of-equilibrium thermostat.
- **Theoretical Guarantees**: Gloeba's attention map is Lipschitz-bounded (Proposition 4), preventing the frozen, gradient-starved attention that constant low-temperature sampling produces.
- **Representational Expansion**: Under weight tying, Gloeba's $M_h$ reaches asymmetric and indefinite compatibility patterns beyond the positive-semidefinite cone that confines standard attention (Proposition 5).
- **Robust Anomaly Detection**: Achieves stable, low-variance AUROC on unsupervised anomaly detection benchmarks (STL-10) under a 90% anomaly-rate regime, using a Ledoit-Wolf-shrunken Mahalanobis distance in the latent space.
- **Parameter Efficiency**: Tied Gloeba achieves higher Tiny ImageNet pretraining accuracy with 4.4% fewer parameters than untied standard attention (59.3M vs. 62.1M parameters).
- **Continuous Representation**: Operates directly on continuous patch embeddings, preserving the spatial geometry lost by binary latent codes in classical EBMs.

---

## Overview
Classical Energy-Based Models (EBMs), such as Restricted Boltzmann Machines (RBMs), learn data distributions via Gibbs or Langevin sampling. A critical flaw is **thermal quenching**: with a fixed sampling temperature in a changing energy landscape, the sampler can freeze in local modes, leading to unstable training and poor gradient estimates. Selyne addresses this by eliminating the sampler entirely.

Instead, Selyne interprets each attention layer as a single, closed-form energy-minimization step on continuous patch embeddings. This is achieved through its core component, Gloeba, which provides a stable, learnable, and non-equilibrium thermodynamic regulation.

### Sample Reconstructions

The following figures show qualitative reconstruction results on STL-10 natural images for both tied Gloeba and untied standard attention.

**Tied Gloeba (Selyne):**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://raw.githubusercontent.com/cagasolu/energy-transformer/main/images/fig_stl10_images_normal.png" alt="STL-10 Normal Reconstructions (Tied)" width="95%">
        <br>
        <em>(a) STL-10 - Normal Samples (Tied Gloeba)</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/cagasolu/energy-transformer/main/images/fig_stl10_images_abnormal.png" alt="STL-10 Anomaly Reconstructions (Tied)" width="95%">
        <br>
        <em>(b) STL-10 - Anomaly Samples (Tied Gloeba)</em>
      </td>
    </tr>
  </table>
</div>

**Untied Standard Attention (Baseline):**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://raw.githubusercontent.com/cagasolu/energy-transformer/main/images/fig_stl10_us_images_normal.png" alt="STL-10 Normal Reconstructions (Untied)" width="95%">
        <br>
        <em>(c) STL-10 - Normal Samples (Untied Standard)</em>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/cagasolu/energy-transformer/main/images/fig_stl10_us_images_abnormal.png" alt="STL-10 Anomaly Reconstructions (Untied)" width="95%">
        <br>
        <em>(d) STL-10 - Anomaly Samples (Untied Standard)</em>
      </td>
    </tr>
  </table>
</div>

---

## Core Concepts & Architecture

### Why Not Traditional EBMs?
Traditional EBMs often suffer from:

- **Binary Bottleneck**: They force continuous visual data into binary latent codes, discarding crucial metric information like position, texture, and geometric proportion.
- **Thermal Quenching**: Fixed-temperature sampling in a non-stationary energy landscape is fundamentally unstable.

Selyne is designed to bypass both of these limitations.

### Selyne Architecture
Selyne is built on a Transformer backbone and consists of:

1. **Patch Embedding**: Splits the input image into patches and projects them into continuous token embeddings.
2. **Gloeba Encoder**: A stack of 6 transformer blocks that utilize the novel Energy-Based Attention.
3. **Prototype Cross-Attention**: Summarizes patch tokens into a fixed set of prototype vectors.
4. **Latent Encoder MLP**: Projects the combined [CLS] token and prototype features into a continuous latent space.
5. **Reconstruction Decoder**: Mirrors the encoder to reconstruct the original image from the latent representation.

### Gloeba: The Core Attention Mechanism
For each head h:

- **Learnable Bilinear Compatibility**: Replaces the standard dot product with a learnable bilinear form:

  $S_h = Q_h M_h K_h^\top$

  Here, $M_h$ is a head-specific, learnable matrix that learns a non-Euclidean similarity geometry.

- **Adaptive Per-Head Temperature**: Uses a learnable temperature $\tau_h = e^{\ell_h}$, clamped to a minimum value $\epsilon = 10^{-4}$. This acts as a thermostat, adaptively controlling the sharpness/softness of the attention and preventing quenching.

- **Lipschitz-Bounded Sensitivity**: The clamp $\tau_h \ge \epsilon$ ensures the attention map is Lipschitz with constant at most $1/(2\epsilon)$, providing a stability guarantee (Proposition 4).

- **Closed-Form Energy Minimization**: The resulting softmax operation is the exact minimizer of an entropy-regularized bilinear energy, providing deterministic stability.

### The Latent Mahalanobis Scoring
After fine-tuning, anomaly detection is performed using two scores:

- **Reconstruction Energy Score**: A pixel-wise reconstruction error (MSE + TV + FFT + pooled error).
- **Mahalanobis Distance (Primary)**: A distance metric in the latent space. A Ledoit-Wolf shrinkage covariance is estimated from the normal training data. The Mahalanobis distance of a new sample is then computed, with anomalies lying further from the normal distribution's mean.

---

## Ablation: Tied Gloeba vs. Untied Standard

We compare tied Gloeba against untied standard attention, the strongest standard variant. The key architectural differences are:

| Component | Untied Standard | Tied Gloeba |
|-----------|-----------------|-------------|
| Q/K Projections | $W_Q, W_K$ independent | $W_Q = W_K = W$ |
| Kernel $A_h$ | $W_Q W_K^\top$ | $W M_h W^\top$ |
| Reachable forms | Full rank-≤$D_h$ | $\mathbb{R}^{D_h \times D_h}$ |
| Temperature | $1/\sqrt{D_h}$ (fixed) | $e^{\ell_h}$ (learnable) |
| Parameters per block | 1,050,624 | 820,752 |
| Total parameters | 62,053,386 | **59,294,922** |

### Pretraining Performance (7 seeds, Tiny ImageNet)

| Metric | Tied Gloeba | Untied Standard | Δ |
|--------|-------------|-----------------|---|
| Best val. accuracy (mean ± std) | 0.5650 ± 0.005 | 0.5594 ± 0.003 | +0.0056 (p=0.011) |
| Best val. accuracy (max) | 0.572 | 0.563 | +0.0090 |
| Total parameters | **59,294,922** | 62,053,386 | -2,758,464 |
| Training time (per run) | 3.3 h | **2.95 h** | +0.35 h |

**Key finding**: Tied Gloeba achieves higher accuracy with 4.4% fewer parameters, at the cost of ~24 additional minutes per run.

### Anomaly Detection Performance (STL-10, 7 seeds)

| Metric | Tied Gloeba | Untied Standard | Δ (p-value) | $d_z$ |
|--------|-------------|-----------------|-------------|-------|
| Mahalanobis AUROC | 0.8948 | **0.8974** | +0.0027 (0.0029) | 1.82 |
| Reconstruction AUROC | 0.5270 | **0.5286** | +0.0016 (0.070) | 0.83 |

**Key finding**: The two variants are practically equivalent detectors. The small aggregate Mahalanobis gap (+0.0027) traces to just two classes pulling opposite ways: Bird favors untied (+0.0410), Ship favors tied (-0.0131). Removing both outliers erases the difference (-0.0002, p=0.71).

### Leave-One-Class-Out Sensitivity

| Removed | Δ (U−T) | p | Favors |
|---------|---------|---|--------|
| None (all 10) | +0.0027 | 0.0029 | Untied (7/7) |
| − Bird | **−0.0016** | 0.0025 | **Tied (7/7)** |
| − Ship | +0.0044 | 0.0006 | Untied (7/7) |
| − both | −0.0002 | 0.709 | n.s. |

**Conclusion**: Tied Gloeba matches the untied standard on detection while pretraining better with fewer parameters. The differences are localized to two outlier classes.

---

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (minimum 8GB VRAM recommended)
- Required libraries (see requirements.txt)

### Installation

Clone the repository:
```bash
git clone https://github.com/cagasolu/energy-transformer.git
cd energy-transformer
