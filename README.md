# energy-transformer

Pretrained weights (~1 GB): https://huggingface.co/cagasoluh/energy-transformer

---

## SELYNE

Selyne (Stable-Energy Lipschitz Net) introduces Gloeba, an energy-based attention mechanism that eliminates MCMC sampling and thermal quenching. Official PyTorch implementation.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20779017.svg)](https://doi.org/10.5281/zenodo.20779017)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/license-GPLv3-blue)

---

## Environment Note

The pretraining scripts are published ready to run on Google systems (Colab / Google Cloud), with Google Drive mounting for saving models and results. The fine-tuning scripts are configured for local GPUs. All file paths are set for these defaults; anyone can freely change the paths to fit their own environment.

---

## Key Highlights

- Removes MCMC sampling from EBMs; closed-form energy minimization
- Gloeba: learnable bilinear compatibility M_h + adaptive temperature tau_h
- Lipschitz-bounded attention (bounded-sensitivity guarantee)
- Expands representable forms beyond the PSD cone (tied non-absorbability)
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

Scoring: Reconstruction (MSE + TV + FFT + pool) + Mahalanobis (Ledoit-Wolf shrinkage)

---

## Ablation: Tied Gloeba vs. Untied Standard

| Component | Untied | Tied Gloeba |
|-----------|--------|-------------|
| $Q/K$ | $W_Q$, $W_K$ | $W_Q$ = $W_K$ = $W$ |
| Kernel | $W_Q W_K^T$ | $W M_h W^T$ |
| Params | $62.1M$ | $59.3M$ |

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

Gap localized to Bird (favors untied) and Ship (favors tied). Removing both -0.0002 and p=0.71 -> practically equivalent detectors.

---

## Quick Start

### Option 1: Google Colab (Recommended for pretraining)

Step-by-step:

    1. Open Google Colab: https://colab.research.google.com/
    2. Go to File -> New Notebook
    3. Run the following cells:

Cell 1 - Clone repository:

    !git clone https://github.com/cagasolu/energy-transformer.git
    %cd energy-transformer

Cell 2 - Install dependencies:

    !pip install -r requirements.txt

Cell 3 - Mount Google Drive (for saving models/results):

    from google.colab import drive
    drive.mount('/content/drive')

    # Create a directory for your runs
    !mkdir -p /content/drive/MyDrive/energy_transformer_results

Cell 4 - Download pretrained weights:

    !pip install huggingface_hub
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="cagasoluh/energy-transformer",
        local_dir="./pretrained_weights",
        local_dir_use_symlinks=False
    )

Cell 5 - Run pretraining:

    # Tied Gloeba pretraining
    !python pretrain_selyne_recon.py

    # Untied standard pretraining (baseline)
    !python pretrain_standard_recon.py

### Option 2: Local Setup (for fine-tuning / anomaly detection)

    git clone https://github.com/cagasolu/energy-transformer.git
    cd energy-transformer
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt

---

## Usage

### Pretraining (Google / Colab)

    # Tied Gloeba (recommended)
    python pretrain_selyne_recon.py

    # Untied standard (baseline)
    python pretrain_standard_recon.py

### STL-10 Anomaly Detection (local GPU)

    # Tied Gloeba
    python globaleba_mahalanobis.py --seed 2584 --gpu 0

    # Untied standard
    python standard_mahalanobis.py --seed 2584 --gpu 0

---

## Google Cloud Setup

For GCP VM with GPU:

    # Create VM with GPU
    gcloud compute instances create energy-transformer-vm \
        --zone us-central1-a \
        --accelerator type=nvidia-tesla-a100 \
        --machine-type a2-highgpu-1g \
        --image-family ubuntu-2204-lts \
        --image-project ubuntu-os-cloud

    # SSH into VM
    gcloud compute ssh energy-transformer-vm

    # Install CUDA and dependencies
    sudo apt update
    sudo apt install python3-pip python3-venv nvidia-driver-535
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # Clone and install
    git clone https://github.com/cagasolu/energy-transformer.git
    cd energy-transformer
    pip install -r requirements.txt

    # Run training with screen (persistent)
    screen -S training
    python pretrain_selyne_recon.py

---

## Results

### STL-10 (90% anomaly)

    Score               | Mean AUROC | CV
    Reconstruction      | 0.5270     | 0.34%
    Mahalanobis         | 0.8948     | 0.14%

Per-class Mahalanobis (example seed 2584):

    Airplane 0.936, Bird 0.784, Car 0.961, Cat 0.848, Deer 0.902, Dog 0.871, Horse 0.912, Monkey 0.880, Ship 0.944, Truck 0.929

Compute: ~43.75 GPU-hours (14 runs, A100)

---

## Pretrained Models

Download from Hugging Face:

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="cagasoluh/energy-transformer",
        local_dir="./pretrained_weights",
        local_dir_use_symlinks=False
    )

The repository hosts the pretrained tied Gloeba and untied standard weights along with their STL-10 fine-tuned checkpoints. See the Hugging Face repo file list for exact filenames before downloading individual files.

---

## File Structure

    energy-transformer/
    ├── pretrain_selyne_recon.py        # Tied Gloeba pretraining (Google / Colab)
    ├── pretrain_standard_recon.py      # Untied standard pretraining (Google / Colab)
    ├── globaleba_mahalanobis.py        # STL-10 anomaly detection, tied (local GPU)
    ├── standard_mahalanobis.py         # STL-10 anomaly detection, untied (local GPU)
    └── requirements.txt

---

## Citation

    @misc{suleymanoglu_selyne,
      author = {Suleymanoglu, Gorkem Can},
      title = {Selyne: Stable-Energy Lipschitz Network with Energy-Based Attention for Anomaly Detection},
      publisher = {GitHub},
      url = {https://github.com/cagasolu/energy-transformer},
      doi = {10.5281/zenodo.20779017}
    }

---

## Links

- **GitHub:** https://github.com/cagasolu/energy-transformer
- **Hugging Face:** https://huggingface.co/cagasoluh/energy-transformer
- **Zenodo:** https://doi.org/10.5281/zenodo.20779017

---

## License

GPLv3
