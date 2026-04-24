# Scalable Neural Pushbroom Architectures for Real-Time Hyperspectral Image Denoising

Official implementation of the paper:

**Scalable neural pushbroom architectures for real-time denoising of hyperspectral images onboard satellites**
*(Accepted by IEEE Transactions on Geoscience and Remote Sensing - TGRS)*

📄 Paper: https://arxiv.org/abs/2601.05020

---

## Overview

This repository provides the official PyTorch implementation of our proposed architecture for hyperspectral image (HSI) denoising in onboard satellite scenarios.

The proposed method addresses three key challenges:

* **Real-time processing under strict computational constraints**
* **Dynamic power scalability**
* **Robustness to radiation-induced faults**

We introduce a **mixture-of-denoisers architecture** with a **line-wise (pushbroom) processing design**, enabling efficient memory usage and real-time inference aligned with hyperspectral acquisition.

---

## Requirements & Environment

This project should be documented from `boost.yml` together with the imports used across the Python source files.

### Base environment from `boost.yml`

- OS: Linux
- Python: `3.10.15`
- PyTorch: `2.5.0`
- CUDA runtime: `11.8` via `pytorch-cuda=11.8`
- TorchVision: `0.20.0`
- NumPy: `2.0.1`

Recommended setup:

```bash
conda env create -f boost.yml
conda activate Denoise
pip install mamba-ssm==2.2.4 --no-build-isolation
pip install causal-conv1d --no-build-isolation
```

### Python packages required by the source code

Besides the base packages already present in `boost.yml`, the codebase imports the following libraries:

- `hydra-core`
- `omegaconf`
- `pytorch-lightning`
- `lightning-bolts`
- `scipy`
- `h5py`
- `lmdb`
- `pyyaml`
- `pillow`
- `imageio`
- `einops`
- `timm`
- `transformers==4.37.2`
- `mamba-ssm==2.2.4`
- `causal-conv1d`
- `torchtyping`
- `typeguard`
- `scikit-image`
- `matplotlib`
- `opencv-python`
- `tqdm`
- `fvcore`
- `thop`
- `torchstat`
- `torchsummary`

### Optional packages for specific scripts

These are only needed for some auxiliary or experimental scripts:

- `nnunetv2` and `dynamic-network-architectures` for `LAMU` and `ssumamba` related model files
- `caffe` for legacy LMDB utility scripts under [`nafmamba/utility`](your_path)

### Compatibility note

`mamba-ssm` and `causal-conv1d` are sensitive to the local CUDA build environment. In this repository, the more reliable installation path is to create the conda environment first and then install them manually with `--no-build-isolation`.

`mamba-ssm==2.2.4` is also sensitive to the `transformers` version. This repository now pins `transformers==4.37.2` in `boost.yml` to avoid the `GreedySearchDecoderOnlyOutput` import error seen with newer releases.

Some legacy utilities such as [dataloaders_hsi.py](your_path) and [dataloaders_hsi_test.py](your_path) still import `torch._six`, which is deprecated in newer PyTorch releases. If those scripts are part of your workflow, they may require a small compatibility patch even when using the environment from `boost.yml`.

---

## Dataset

We use the ICVL hyperspectral dataset:

👉 https://huggingface.co/datasets/danaroth/icvl

Please follow the dataset instructions and place the data in the appropriate directory (to be specified).

---

## Training

Example training command:

```bash
sh train95_ensemblenafmamba
```

---

## Testing

Example testing command:

```bash
sh testmix.sh
```

## Reproducibility

We provide scripts to reproduce key results:
- Power scalability experiments
- Fault tolerance evaluation 
python 

---

## Pretrained Models

Pretrained models are avaiable [here](https://www.dropbox.com/scl/fo/kj0de53tv2zuu9sbpncxg/AAhZqDko1cyHROTRl1moEzw?rlkey=e96223lh03yfmq765797xhwyb&st=gw0hrzxh&dl=0)



## Code Base

This implementation is **strongly based on**:

👉 https://github.com/lronkitty/SSRT

We thank the authors for their excellent work.

---

## Acknowledgements

This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013, CIG B421A95680, CUP E13C22001800001). 



