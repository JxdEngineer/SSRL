# %% import libraries
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import numpy as np
import torch
import importlib

import matplotlib.pyplot as plt

from scipy import stats, signal
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score, accuracy_score

from scripts.artificial_features_MCC5.configs import dataset_splits_path
# %% process data
# load data
splits = torch.load(dataset_splits_path, map_location="cpu")

acc_tr, exc_tr, dmg_tr = splits["acc_tr"],splits["exc_tr"], splits["dmg_tr"]
acc_va, exc_va, dmg_va = splits["acc_va"],splits["exc_va"], splits["dmg_va"]
acc_te, exc_te, dmg_te = splits["acc_te"],splits["exc_te"], splits["dmg_te"]

acc_all = torch.cat([acc_tr, acc_va, acc_te], dim=0)  # (N, 12, K)
exc_all = torch.cat([exc_tr, exc_va, exc_te], dim=0).view(-1)  # (N,)
dmg_all = torch.cat([dmg_tr, dmg_va, dmg_te], dim=0).view(-1)  # (N,)

# convert to numpy
acc_all = acc_all.detach().cpu().numpy()            # (N, C, K)
exc_all = exc_all.detach().cpu().numpy().astype(int).reshape(-1)
dmg_all = dmg_all.detach().cpu().numpy().astype(int).reshape(-1)

print("acc_all:", acc_all.shape, "exc unique:", np.unique(exc_all), "dmg unique:", np.unique(dmg_all))

# %%
# %% (optional) filter excitations like your script
exc_labels = [3]  # change if needed
exc_mask = np.isin(exc_all, np.array(exc_labels, dtype=int))

X = acc_all[exc_mask]
exc = exc_all[exc_mask]
dmg = dmg_all[exc_mask]

print("selected exc:", exc_labels, "count:", X.shape[0])

# %% baseline mask: 5 May (Europe/Zurich) whole day
base_mask = (dmg == 1)

print("baseline count:", int(base_mask.sum()), "baseline damaged count:", int((dmg[base_mask] != 1).sum()))

# %% compute artificial features (library-based, no custom feature functions from scratch)
# Features per channel computed over K dimension (frequency bins):
# mean, std, skew, kurtosis, max, argmax_norm, entropy, spectral_centroid_norm, bandpower (sum)
eps = 1e-12
N, C, K = X.shape

# Precompute a frequency axis in [0,1] just to compute centroid-like summaries
freq = np.linspace(0.0, 1.0, K, dtype=np.float64)

# basic stats from scipy
mean_ = X.mean(axis=2)                          # (N,C)
std_  = X.std(axis=2) + eps                     # (N,C)
skew_ = stats.skew(X, axis=2, bias=False, nan_policy="omit")     # (N,C)
kurt_ = stats.kurtosis(X, axis=2, bias=False, nan_policy="omit") # (N,C)
max_  = X.max(axis=2)                           # (N,C)
argm_ = X.argmax(axis=2).astype(np.float64) / max(1, (K - 1))    # (N,C) normalized

# entropy needs nonnegative and sum-to-1 along K
p = np.clip(X, 0.0, None)
p = p / (p.sum(axis=2, keepdims=True) + eps)
ent_ = stats.entropy(p + eps, axis=2)           # (N,C)

# spectral centroid and bandpower using scipy.signal helpers
# centroid = sum(f * P) / sum(P)
cent_ = (p * freq[None, None, :]).sum(axis=2) / (p.sum(axis=2) + eps)  # (N,C)
bp_   = p.sum(axis=2)                           # (N,C) should be 1, but keep for completeness

# Stack features: shape (N, C*F)
feat = np.stack([mean_, std_, skew_, kurt_, max_, argm_, ent_, cent_, bp_], axis=-1)  # (N,C,F)
feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
F = feat.shape[-1]
feat_all = feat.reshape(N, C * F)

print("feature matrix:", feat_all.shape)
# %%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# latent feature visualization, 2D UMAP
import utils.plot_latent as pl
importlib.reload(pl)

plt.close()

pl.plot_latent_2d(feat_all, "artificial feature", dmg, exc)
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Unsupervised damage detection on z_dmg (one-class novelty on 5 May baseline)
from utils.calculate_damage_score import plot_mahal
percentile = 95

mahal, base_mask, _ = plot_mahal(feat_all, exc, dmg, p=percentile, name="artificial features", healthy_label=1,
                                          per_exc_baseline=False, enable_plot=True)
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Binary classification using baseline p95 threshold on Mahalanobis distance
from utils.damage_detection import eval_damage_by_percentile

print("involved excitation labels:", exc_labels)

cm = eval_damage_by_percentile(mahal, base_mask, dmg, percentile, healthy_label=1, name="artificial features mahal")