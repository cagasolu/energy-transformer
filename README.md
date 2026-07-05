# energy-transformer

The pretrained model weights are available on Hugging Face due to their large size (~750 MB).

Download them from the repository below:

Hugging Face: https://huggingface.co/cagasoluh/energy-transformer


## SELYNE: Stable-Energy Lipschitz Network with Energy-Based Attention for Anomaly Detection

Selyne (Stable-Energy Lipschitz Net) introduces a novel energy-based attention mechanism, Gloeba (Global Energy-Based Attention), designed to overcome the thermal-quenching instability observed in traditional energy-based models. This repository contains the official PyTorch implementation, pretrained models, and evaluation code.

DOI: 10.5281/zenodo.20779017

Python 3.10+ | PyTorch 2.0+ | Linux | CUDA 11.8+ | Google Cloud | Colab | GPLv3


## Table of Contents

- Key Highlights
- Overview
- Core Concepts and Architecture
- Ablation: Tied Gloeba vs. Untied Standard
- Getting Started
  - Google Cloud / Colab Setup
  - Local Setup
- Usage
  - Pretraining on Tiny ImageNet
  - Anomaly Detection on STL-10
  - Anomaly Detection on BRISC2025 Brain MRI
- Results Summary
- License
- Citation
- Acknowledgements


## Key Highlights

- Addresses Thermal Quenching: Selyne removes the unstable MCMC sampling step inherent in classical Energy-Based Models, replacing it with a deterministic, closed-form energy minimization process.

- Novel Attention Mechanism: Introduces Gloeba (Global Energy-Based Attention), which uses a learnable, per-head bilinear compatibility matrix (M_h) and an adaptive, per-head temperature (tau_h) as an out-of-equilibrium thermostat.

- Theoretical Guarantees: Gloeba's attention map is Lipschitz-bounded (Proposition 4), preventing the frozen, gradient-starved attention that constant low-temperature sampling produces.

- Representational Expansion: Under weight tying, Gloeba's M_h reaches asymmetric and indefinite compatibility patterns beyond the positive-semidefinite cone that confines standard attention (Proposition 5).

- Robust Anomaly Detection: Achieves stable, low-variance AUROC on unsupervised anomaly detection benchmarks (STL-10) under a 90% anomaly-rate regime, using a Ledoit-Wolf-shrunken Mahalanobis distance in the latent space.

- Parameter Efficiency: Tied Gloeba achieves higher Tiny ImageNet pretraining accuracy with 4.4% fewer parameters than untied standard attention (59.3M vs. 62.1M parameters).

- Continuous Representation: Operates directly on continuous patch embeddings, preserving the spatial geometry lost by binary latent codes in classical EBMs.

- Cloud-Ready: Pretraining scripts are optimized for Google Cloud and Google Colab with automatic Google Drive mounting and persistent worker support.


## Overview

Classical Energy-Based Models (EBMs), such as Restricted Boltzmann Machines (RBMs), learn data distributions via Gibbs or Langevin sampling. A critical flaw is thermal quenching: with a fixed sampling temperature in a changing energy landscape, the sampler can freeze in local modes, leading to unstable training and poor gradient estimates. Selyne addresses this by eliminating the sampler entirely.

Instead, Selyne interprets each attention layer as a single, closed-form energy-minimization step on continuous patch embeddings. This is achieved through its core component, Gloeba, which provides a stable, learnable, and non-equilibrium thermodynamic regulation.

### Sample Reconstructions

The following figures show qualitative reconstruction results on STL-10 natural images for both tied Gloeba and untied standard attention.

Tied Gloeba (Selyne):

(a) STL-10 - Normal Samples (Tied Gloeba)
https://raw.githubusercontent.com/cagasolu/energy-transformer/main/images/fig_stl10_images_normal.png

(b) STL-10 - Anomaly Samples (Tied Gloeba)
https://raw.githubusercontent.com/cagasolu/energy-transformer/main/images/fig_stl10_images_abnormal.png

Untied Standard Attention (Baseline):

(c) STL-10 - Normal Samples (Untied Standard)
https://raw.githubusercontent.com/cagasolu/energy-transformer/main/images/fig_stl10_us_images_normal.png

(d) STL-10 - Anomaly Samples (Untied Standard)
https://raw.githubusercontent.com/cagasolu/energy-transformer/main/images/fig_stl10_us_images_abnormal.png


## Core Concepts and Architecture

### Why Not Traditional EBMs?

Traditional EBMs often suffer from:

- Binary Bottleneck: They force continuous visual data into binary latent codes, discarding crucial metric information like position, texture, and geometric proportion.

- Thermal Quenching: Fixed-temperature sampling in a non-stationary energy landscape is fundamentally unstable.

Selyne is designed to bypass both of these limitations.

### Selyne Architecture

Selyne is built on a Transformer backbone and consists of:

1. Patch Embedding: Splits the input image into patches and projects them into continuous token embeddings.

2. Gloeba Encoder: A stack of 6 transformer blocks that utilize the novel Energy-Based Attention.

3. Prototype Cross-Attention: Summarizes patch tokens into a fixed set of prototype vectors.

4. Latent Encoder MLP: Projects the combined [CLS] token and prototype features into a continuous latent space.

5. Reconstruction Decoder: Mirrors the encoder to reconstruct the original image from the latent representation.

### Gloeba: The Core Attention Mechanism

For each head h:

- Learnable Bilinear Compatibility: Replaces the standard dot product with a learnable bilinear form:

  S_h = Q_h M_h K_h^T

  Here, M_h is a head-specific, learnable matrix that learns a non-Euclidean similarity geometry.

- Adaptive Per-Head Temperature: Uses a learnable temperature tau_h = e^(ell_h), clamped to a minimum value epsilon = 1e-4. This acts as a thermostat, adaptively controlling the sharpness/softness of the attention and preventing quenching.

- Lipschitz-Bounded Sensitivity: The clamp tau_h >= epsilon ensures the attention map is Lipschitz with constant at most 1/(2*epsilon), providing a stability guarantee (Proposition 4).

- Closed-Form Energy Minimization: The resulting softmax operation is the exact minimizer of an entropy-regularized bilinear energy, providing deterministic stability.

### The Latent Mahalanobis Scoring

After fine-tuning, anomaly detection is performed using two scores:

- Reconstruction Energy Score: A pixel-wise reconstruction error (MSE + TV + FFT + pooled error).

- Mahalanobis Distance (Primary): A distance metric in the latent space. A Ledoit-Wolf shrinkage covariance is estimated from the normal training data. The Mahalanobis distance of a new sample is then computed, with anomalies lying further from the normal distribution's mean.


## Ablation: Tied Gloeba vs. Untied Standard

We compare tied Gloeba against untied standard attention, the strongest standard variant. The key architectural differences are:

Component           | Untied Standard     | Tied Gloeba
--------------------|---------------------|------------
Q/K Projections     | W_Q, W_K independent| W_Q = W_K = W
Kernel A_h          | W_Q W_K^T           | W M_h W^T
Reachable forms     | Full rank <= D_h    | R^(D_h x D_h)
Temperature         | 1/sqrt(D_h) (fixed) | e^(ell_h) (learnable)
Parameters per block| 1,050,624           | 820,752
Total parameters    | 62,053,386          | 59,294,922

### Pretraining Performance (7 seeds, Tiny ImageNet)

Metric                          | Tied Gloeba        | Untied Standard    | Delta
--------------------------------|--------------------|--------------------|-----------
Best val. accuracy (mean +/- sd)| 0.5650 +/- 0.005   | 0.5594 +/- 0.003   | +0.0056 (p=0.011)
Best val. accuracy (max)        | 0.572              | 0.563              | +0.0090
Total parameters                | 59,294,922         | 62,053,386         | -2,758,464
Training time (per run)         | 3.3 h              | 2.95 h             | +0.35 h

Key finding: Tied Gloeba achieves higher accuracy with 4.4% fewer parameters, at the cost of ~24 additional minutes per run.

### Anomaly Detection Performance (STL-10, 7 seeds)

Metric              | Tied Gloeba | Untied Standard | Delta (p-value) | d_z
--------------------|-------------|-----------------|-----------------|-----
Mahalanobis AUROC   | 0.8948      | 0.8974          | +0.0027 (0.0029)| 1.82
Reconstruction AUROC| 0.5270      | 0.5286          | +0.0016 (0.070) | 0.83

Key finding: The two variants are practically equivalent detectors. The small aggregate Mahalanobis gap (+0.0027) traces to just two classes pulling opposite ways: Bird favors untied (+0.0410), Ship favors tied (-0.0131). Removing both outliers erases the difference (-0.0002, p=0.71).

### Leave-One-Class-Out Sensitivity

Removed         | Delta (U-T) | p     | Favors
----------------|-------------|-------|-----------
None (all 10)   | +0.0027     | 0.0029| Untied (7/7)
- Bird          | -0.0016     | 0.0025| Tied (7/7)
- Ship          | +0.0044     | 0.0006| Untied (7/7)
- both          | -0.0002     | 0.709 | n.s.

Conclusion: Tied Gloeba matches the untied standard on detection while pretraining better with fewer parameters. The differences are localized to two outlier classes.


## Getting Started

### Google Cloud / Colab Setup

The pretraining scripts (pretrain_selyne_recon.py and pretrain_standard_recon.py) are optimized for Google Cloud and Google Colab environments:

- Automatic Google Drive Mounting: pretrain_selyne_recon.py automatically mounts Google Drive and saves checkpoints to /content/drive/MyDrive/. This ensures model weights persist across sessions.

- Persistent Workers: DataLoaders use persistent_workers=True and pin_memory=True for efficient GPU utilization.

- A100 GPU Support: All experiments were conducted on NVIDIA A100-SXM4-40GB GPUs via Google Colab Enterprise and Google Cloud.

- High-RAM Mode: Recommended to enable high-RAM mode in Colab for optimal performance with the 59.3M-parameter model.

Quick Start on Colab:

# In a Colab notebook:
!git clone https://github.com/cagasolu/energy-transformer.git
%cd energy-transformer
!pip install -r requirements.txt

# Mount Google Drive (automatically handled by script)
from google.colab import drive
drive.mount('/content/drive')

# Run pretraining
!python pretrain_selyne_recon.py

# Or run evaluation
!python globaleba_mahalanobis.py

### Local Setup

Prerequisites:

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (minimum 8GB VRAM recommended)
- Required libraries (see requirements.txt)

Installation:

Clone the repository:

git clone https://github.com/cagasolu/energy-transformer.git
cd energy-transformer

Set up a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

### Dataset Preparation

- Tiny ImageNet: The script pretrain_selyne_recon.py will automatically download and use the dataset. You can also place it in a ./data directory.

- STL-10: The globaleba_mahalanobis.py script will download it automatically via torchvision.


## Usage

Important: Before running any script, ensure you have downloaded the pretrained model files (.pt files) from the Hugging Face repository or Zenodo and placed them in the project root directory.

### 1. Pretraining on Tiny ImageNet

Trains the full Selyne model from scratch on Tiny ImageNet classification.

python pretrain_selyne_recon.py

- Runs for up to 80 epochs with early stopping (patience 20).

- Saves pretrained_selyne_full_recon_best.pt and pretrained_selyne_full_recon.pt.

- On Colab/Cloud: Checkpoints are saved to Google Drive at /content/drive/MyDrive/.

- Generates four professional training visualizations:
  - fig1_loss_dashboard.png (4-panel loss)
  - fig2_accuracy_lr.png (Accuracy + LR schedule)
  - fig3_loss_contribution.png (Loss component ratios)
  - fig4_convergence.png (Convergence + gradient analysis)

For Untied Standard Baseline:

python pretrain_standard_recon.py

### 2. Anomaly Detection on STL-10

Evaluates the model on unsupervised anomaly detection (10 classes, 90% anomaly rate).

python globaleba_mahalanobis.py

- Loads the pretrained checkpoint.

- Fine-tunes for each class (10 epochs, batch size 32).

- Computes both Energy and Mahalanobis AUROC scores.

- Saves results:
  - AUROC bar charts
  - ROC curves with +/- 1 std bands
  - Score distributions (normal vs anomaly)
  - Sample reconstructions with PSNR metrics

For Untied Standard Baseline:

python standard_mahal_science.py

### 3. Anomaly Detection on BRISC2025 Brain MRI

Evaluates the model's ability to detect brain tumors (no_tumor normal, all tumor types anomalous).

python energetic_mahal_science.py

- Loads the pretrained checkpoint.

- Fine-tunes the model.

- Computes Energy and Mahalanobis AUROC scores.

- Saves side-by-side reconstructions with difference maps.


## Results Summary

### STL-10 Anomaly Detection (90% Anomaly Rate)

Score Type                  | Mean AUROC | Std    | CV
----------------------------|------------|--------|--------
Reconstruction (Energy)     | 0.5270     | 0.0018 | 0.341%
Mahalanobis (Latent)        | 0.8948     | 0.0012 | 0.138%

Per-class Mahalanobis AUROC (seed 2584):

Class       | AUROC | Class    | AUROC
------------|-------|----------|-------
Airplane    | 0.9360| Horse    | 0.9120
Bird        | 0.7835| Monkey   | 0.8802
Car         | 0.9614| Ship     | 0.9437
Cat         | 0.8483| Truck    | 0.9286
Deer        | 0.9023| Mean     | 0.8967
Dog         | 0.8710| Std      | 0.054

### BRISC2025 Brain MRI (86% Anomaly Rate)

Score Type                  | Mean AUROC | Std    | CV
----------------------------|------------|--------|--------
Reconstruction (Energy)     | 0.753      | 0.019  | 2.56%
Mahalanobis (Latent)        | 0.852      | 0.015  | 1.71%

### Pretraining Summary (Tiny ImageNet, 7 seeds)

Metric                          | Tied Gloeba | Untied Standard
--------------------------------|-------------|-----------------
Best val. accuracy (mean)       | 0.5650      | 0.5594
Best val. accuracy (max)        | 0.572       | 0.563
Total parameters                | 59,294,922  | 62,053,386
Training time (Colab A100)      | 3.3 h       | 2.95 h

### Computational Resources

All experiments were conducted using:

- Google Colab Enterprise: NVIDIA A100-SXM4-40GB GPUs

- Google Cloud: Compute-optimized instances with high-bandwidth memory and NVLink interconnects

- Total compute: ~43.75 GPU-hours across 14 runs (7 tied Gloeba + 7 untied standard)

- Per-run time: 2.95-3.3 hours for 80 epochs on Tiny ImageNet


## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.


## Citation

If you find our work useful for your research, please consider citing it:

@misc{suleymanoglu2026selyne,
  author       = {Suleymanoglu, Gorkem Can},
  title        = {Selyne: Stable-Energy Lipschitz Network with Energy-Based Attention for Anomaly Detection},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/cagasolu/energy-transformer}},
  doi          = {10.5281/zenodo.20779017}
}


## Acknowledgements

This work was made possible by the computing resources and support from Kuanka Publishing LLC, Google Cloud, and Google Colab Enterprise. The code is available on:

- GitHub: https://github.com/cagasolu/energy-transformer

- Hugging Face: https://huggingface.co/cagasoluh/energy-transformer

- Zenodo: https://doi.org/10.5281/zenodo.20779017

Note: For the most up-to-date information, please refer to the code repository directly.
