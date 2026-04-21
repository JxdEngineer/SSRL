# SSRL

Official repository for the paper:

**Disentangling Damage from Operational Variability: A Label-Free Self-Supervised Representation Learning Framework for Output-Only Structural Damage Identification**

## Overview

This repository contains the implementation of the self-supervised representation learning framework proposed in the paper, with the current public code focused on the **MCC5 gearbox dataset**.

The method learns two latent representations from raw vibration signals:

- `z_dmg`: damage-sensitive representation
- `z_ndmg`: nuisance-related representation

The model is trained with:

- time-domain reconstruction loss
- PSD reconstruction loss
- self-supervised VICReg loss on baseline healthy samples

Damage identification is then performed using **Mahalanobis distance** in the learned latent space.

## Demonstratiev Data
Download the processed MCC5 dataset from: https://drive.google.com/file/d/1vgFMbcAKVf_FN38JXinLZzP-fRx1NbYa/view?usp=drive_link
Then place the file here: data/MCC5.pt

## Train
The default config is stored in: scripts/timeseries_MCC5/configs.py
Run training with: python scripts/timeseries_MCC5/train.py

## Evaluate
Run evaluation with: python scripts/timeseries_MCC5/test_v2.py

## Ablation study
Run ablation study with: python scripts/timeseries_MCC5/run_sweep_all.py

## Handcrafted-feature baseline
An optional handcrafted-feature baseline is provided in: scripts/artificial_features_MCC5/evaluation_v1.py

## Citation

If you use this repository, please cite:

@article{jian2026ssrl,
  title={Disentangling Damage from Operational Variability: A Label-Free Self-Supervised Representation Learning Framework for Output-Only Structural Damage Identification},
  author={Jian, Xudong and Stoura, Charikleia and Scandella, Simon and Chatzi, Eleni},
  journal={To be assigned},
  year={2026}
}

## Contact
Xudong Jian
ETH Zurich
xudong.jian@ibk.baug.ethz.ch
