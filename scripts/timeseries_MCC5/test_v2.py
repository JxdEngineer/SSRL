# v2: only use training set from 5 May as the baseline subset to compute M distance threshold
# %% import libraries
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import matplotlib as mpl

import importlib

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from models.autoencoder import TwoHeadAutoEncoder1D
from models.mlp import PSDHead

from utils.save_model import load_checkpoint

from scripts.timeseries_MCC5.configs import (
    dataset_splits_path,
    results_dir,
    batch_size,
    z_ndmg_ch,
    z_dmg_ch,
    h_dim,
    ckpt_name,
    psd_nfft,
    psd_to_db,           # True => 10*log10(PSD + eps)
    psd_eps,
    psd_K,
)
# %% process data
# load data
splits = torch.load(root/dataset_splits_path, map_location="cpu")

acc_tr, psd_tr, exc_tr, dmg_tr= splits["acc_tr"], splits["psd_tr"],splits["exc_tr"], splits["dmg_tr"]
acc_va, psd_va, exc_va, dmg_va= splits["acc_va"], splits["psd_va"],splits["exc_va"], splits["dmg_va"]
acc_te, psd_te, exc_te, dmg_te= splits["acc_te"], splits["psd_te"],splits["exc_te"], splits["dmg_te"]

# evaluation on training set
# acc_all = acc_tr   # (N, 12, L)
# psd_all = psd_tr   # (N, 12, K)
# exc_all = exc_tr   # (N,) or (N,1)
# dmg_all = dmg_tr   # (N,) or (N,1)

# evaluation on validaiton and testing set
acc_all = torch.cat([acc_va, acc_te], dim=0)   # (N, 12, L)
psd_all = torch.cat([psd_va, psd_te], dim=0)   # (N, 12, K)
exc_all = torch.cat([exc_va, exc_te], dim=0)   # (N,) or (N,1)
dmg_all = torch.cat([dmg_va, dmg_te], dim=0)   # (N,) or (N,1)

# evaluation on all data
# acc_all = torch.cat([acc_tr, acc_va, acc_te], dim=0)   # (N, 12, L)
# psd_all = torch.cat([psd_tr, psd_va, psd_te], dim=0)   # (N, 12, K)
# exc_all = torch.cat([exc_tr, exc_va, exc_te], dim=0)   # (N,) or (N,1)
# dmg_all = torch.cat([dmg_tr, dmg_va, dmg_te], dim=0)   # (N,) or (N,1)

# flatten labels to 1D for masking
exc_all_1d = exc_all.view(-1)
dmg_all_1d = dmg_all.view(-1)

exc_labels = [1,2,3]  # choose multiple
exc_mask = torch.isin(exc_all_1d, torch.tensor(exc_labels))

print("acc_all:", acc_all.shape)
print("psd_all:", psd_all.shape)
print("selected for exc_label:", exc_labels, "count:", int(exc_mask.sum().item()))

# dataset + loader
ds = TensorDataset(
    acc_all[exc_mask],
    psd_all[exc_mask],
    exc_all_1d[exc_mask],
    dmg_all_1d[exc_mask],
)
data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

exc_tr_1d = exc_tr.view(-1)
dmg_tr_1d = dmg_tr.view(-1)
exc_mask_tr = torch.isin(exc_tr_1d, torch.tensor(exc_labels))
# use to get the threshold of Mahalanobis distances from self-supervised training
ds_tr = TensorDataset(
    acc_tr[exc_mask_tr],
    psd_tr[exc_mask_tr],
    exc_tr_1d[exc_mask_tr],
    dmg_tr_1d[exc_mask_tr],
)
data_loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

ckpt_path = str(root / results_dir / ckpt_name)
print("ckpt:", ckpt_path)

# ===== convert to numpy for simplicity =====
exc_np = exc_all_1d[exc_mask].cpu().numpy()
dmg_np = dmg_all_1d[exc_mask].cpu().numpy()

# ===== count per excitation =====
print("\nCount per excitation:")
for e in sorted(set(exc_np)):
    print(f"exc {int(e)}:", (exc_np == e).sum())

# ===== count per damage =====
print("\nCount per damage:")
for d in sorted(set(dmg_np)):
    print(f"dmg {int(d)}:", (dmg_np == d).sum())

# ===== count per (exc, dmg) =====
print("\nCount per (exc, dmg):")
for d in sorted(set(dmg_np)):
    for e in sorted(set(exc_np)):
        count = ((exc_np == e) & (dmg_np == d)).sum()
        if count > 0:
            print(f"exc {int(e)}, dmg {int(d)}: {count}")
# %%
# model inference
# build model exactly like training
in_ch = acc_tr.shape[1]   # 12
model = TwoHeadAutoEncoder1D(in_ch=in_ch, z_dmg_ch=z_dmg_ch, z_ndmg_ch=z_ndmg_ch).to(device)
load_checkpoint(ckpt_path, model, map_location=device)

psd_head = PSDHead(in_dim=z_dmg_ch, out_ch=in_ch, nfft=psd_nfft, hidden=h_dim).to(device)
psd_head_ckpt_path = str(ckpt_path).replace(".pt", "_psd_head.pt")
psd_head_ckpt = torch.load(psd_head_ckpt_path, map_location=device)
psd_head.load_state_dict(psd_head_ckpt["psd_head"])

model.eval()
psd_head.eval()

def collect_embeddings(loader):
    xs, x_hats, psds, psd_hats, z_ndmgs, z_dmgs, excs, dmgs = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for x, psd, exc, dmg in loader:
            x = x.to(device)

            x_hat, z_ndmg, z_dmg = model(x)

            psd_hat = psd_head(z_dmg.mean(dim=2))
            psd_hat = psd_hat[:, :, :psd_K]
            if psd_to_db == True:
                psd = 10 * torch.log10(psd.to(device) + psd_eps)
                psd = psd[:, :, :psd_K]
                psd = psd - torch.mean(psd, dim=(1, 2), keepdim=True)
            else:
                psd = psd.to(device)
                psd = psd[:, :, :psd_K]
                psd = psd - torch.mean(psd, dim=(1, 2), keepdim=True)

            xs.append(x.detach().cpu().numpy())
            x_hats.append(x_hat.detach().cpu().numpy())

            psds.append(psd.detach().cpu().numpy())
            psd_hats.append(psd_hat.detach().cpu().numpy())

            z_ndmgs.append(z_ndmg.detach().cpu().numpy())
            z_dmgs.append(z_dmg.detach().cpu().numpy())

            excs.append(exc.detach().cpu().numpy())
            dmgs.append(dmg.detach().cpu().numpy())

    x_all = np.concatenate(xs, axis=0)
    x_hat_all = np.concatenate(x_hats, axis=0)
    psd_all = np.concatenate(psds, axis=0)
    psd_hat_all = np.concatenate(psd_hats, axis=0)
    z_ndmg_all = np.concatenate(z_ndmgs, axis=0)
    z_dmg_all = np.concatenate(z_dmgs, axis=0)
    exc_all_np = np.concatenate(excs, axis=0).reshape(-1)
    dmg_all_np = np.concatenate(dmgs, axis=0).reshape(-1)

    return x_all, x_hat_all, psd_all, psd_hat_all, z_ndmg_all, z_dmg_all, exc_all_np, dmg_all_np

x, x_hat, psd, psd_hat, z_ndmg, z_dmg, exc, dmg= collect_embeddings(data_loader)
_, _, _, _, z_ndmg_tr, z_dmg_tr, _, dmg_tr= collect_embeddings(data_loader_tr)

print("input x:", x.shape)
print("reconstruction x_hat:", x_hat.shape)
print("input psd:", psd.shape)
print("reconstruction psd_hat:", psd_hat.shape)
print("embeddings z_ndmg:", z_ndmg.shape)
print("embeddings z_dmg:", z_dmg.shape)
print("exc labels:", exc.shape, "unique:", np.unique(exc))
print("damage labels:", dmg.shape, "unique:", np.unique(dmg))

print("embeddings z_ndmg_tr:", z_ndmg_tr.shape)
print("embeddings z_dmg_tr:", z_dmg_tr.shape)

batch_no = 1
print(f"Excitation {exc[batch_no]}, damage {dmg[batch_no]} normalized signals")
mpl.rcParams.update({"font.size": 10.5})
# quick visualization of reconstructed x_hat
fig, axes = plt.subplots(2, 3, figsize=(6, 3), sharex=True)
axes = axes.ravel()
for ch in range(6):
    axes[ch].plot(x[batch_no, ch, :200], linewidth=1.25, label="true")
    axes[ch].plot(x_hat[batch_no, ch, :200], linewidth=1, linestyle="--", label="recon")
    axes[ch].set_title(f"Channel {ch+1}", )
    axes[ch].grid(True)
    axes[ch].set_ylim(-5, 5)
    if ch == 0:
        handles, labels = axes[ch].get_legend_handles_labels()
        axes[ch].legend(handles, labels,
                    ncol=1, #len(labels),   # all items in one row
                    fontsize=9,
                    loc='lower right')
# fig.suptitle(f"Excitation {exc[batch_no]}, normalized signals", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# quick visualization of reconstructed psd_hat
fig, axes = plt.subplots(2, 3, figsize=(6, 3), sharex=True)
axes = axes.ravel()
for ch in range(6):
    axes[ch].plot(psd[batch_no, ch, :200], linewidth=1, label="true")
    axes[ch].plot(psd_hat[batch_no, ch, :200], linewidth=1, linestyle="--", label="recon")
    axes[ch].set_title(f"Channel {ch+1}")
    axes[ch].grid(True)
    axes[ch].set_ylim(-25, 35)
    if ch == 0:
        handles, labels = axes[ch].get_legend_handles_labels()
        axes[ch].legend(handles, labels,
                    ncol=1, #len(labels),   # all items in one row
                    fontsize=9,
                    loc='lower right')
# fig.suptitle(f"Excitation {exc[batch_no]}, normalized PSDs", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# %%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# latent feature visualization, 2D UMAP
import utils.plot_latent as pl
importlib.reload(pl)

plt.close()

pl.plot_latent_2d(z_dmg, "z_dmg", dmg, exc)
pl.plot_latent_2d(z_ndmg, "z_ndmg", dmg, exc)

# plot_latent_3d(z_dmg, "z_dmg", dmg, exc, labels_dt=labels_dt)
# plot_latent_3d(z_ndmg, "z_ndmg", dmg, exc, labels_dt=labels_dt)

# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Unsupervised damage detection on z_dmg (one-class novelty on 5 May baseline) - old version, use M dist from all samples on 5 May to compute threshold

import utils.calculate_damage_score as cds
importlib.reload(cds)

percentile = 95

mahal_dmg, thr_dmg = cds.plot_mahal_baseline_train(z_dmg, z_dmg_tr, exc, dmg, dmg_tr,
                                                          p=percentile, name="z_dmg", 
                                                          enable_plot=True)
mahal_ndmg, thr_ndmg = cds.plot_mahal_baseline_train(z_ndmg, z_ndmg_tr, exc, dmg, dmg_tr,
                                                          p=percentile, name="z_ndmg", 
                                                          enable_plot=True)
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Binary classification using baseline p95 threshold on Mahalanobis distance
from utils.damage_detection import eval_damage_by_percentile_baseline_train

print("involved excitation labels:", exc_labels)

cm_dmg = eval_damage_by_percentile_baseline_train(mahal_dmg, thr_dmg, dmg, percentile, healthy_label=1, name="z_dmg mahal")
cm_ndmg = eval_damage_by_percentile_baseline_train(mahal_ndmg, thr_ndmg, dmg, percentile, healthy_label=1, name="z_ndmg mahal")
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute confusion matrace of different conditions
exc_labels = [2]  # choose less
# exc_labels = [1,2,3]  # choose all

exc_mask = torch.isin(exc_all_1d, torch.tensor(exc_labels))
# dataset + loader
ds = TensorDataset(
    acc_all[exc_mask],
    psd_all[exc_mask],
    exc_all_1d[exc_mask],
    dmg_all_1d[exc_mask],
)
data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
x, x_hat, psd, psd_hat, z_ndmg, z_dmg, exc, dmg = collect_embeddings(data_loader)

exc_mask_tr = torch.isin(exc_tr_1d, torch.tensor(exc_labels))
# use to get the threshold of Mahalanobis distances from self-supervised training
ds_tr = TensorDataset(
    acc_tr[exc_mask_tr],
    psd_tr[exc_mask_tr],
    exc_tr_1d[exc_mask_tr],
    dmg_tr_1d[exc_mask_tr],
)
data_loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False)
_, _, _, _, z_ndmg_tr, z_dmg_tr, _, dmg_tr= collect_embeddings(data_loader_tr)

mahal_dmg, thr_dmg = cds.plot_mahal_baseline_train(z_dmg, z_dmg_tr, exc, dmg, dmg_tr,
                                                          p=percentile, name="z_dmg", 
                                                          enable_plot=True)
mahal_ndmg, thr_ndmg = cds.plot_mahal_baseline_train(z_ndmg, z_ndmg_tr, exc, dmg, dmg_tr,
                                                          p=percentile, name="z_ndmg", 
                                                          enable_plot=True)

print("involved excitation labels:", exc_labels)

cm_dmg = eval_damage_by_percentile_baseline_train(mahal_dmg, thr_dmg, dmg, percentile, healthy_label=1, name="z_dmg mahal")
cm_ndmg = eval_damage_by_percentile_baseline_train(mahal_ndmg, thr_ndmg, dmg, percentile, healthy_label=1, name="z_ndmg mahal")

# plot hidden features
# sec_per_day = 24 * 3600
# d0 = datetime(2025, 5, 5, 0, 0, 0, tzinfo=timezone.utc).timestamp()
# day_idx = np.floor(((dt - d0) // sec_per_day)).astype(int)
# day_map = {0: "5 May", 1: "6 May", 2: "7 May"}
# labels_dt = np.array([day_map.get(i, "other") for i in day_idx])
# plot_latent_2d(z_dmg, "z_dmg", dmg, exc, labels_dt=labels_dt)
# plot_latent_2d(z_ndmg, "z_ndmg", dmg, exc, labels_dt=labels_dt)

# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Record results
exc_sets = [
    [1],
    [2],
    [3],
    [1, 2, 3],
]

cm_row_labels = ["TN", "FP", "FN", "TP"]  # for printing only

cm_blocks_dmg = []
cm_blocks_ndmg = []

for exc_labels in exc_sets:
    print("\t","==========================")
    print("involved excitation labels:", exc_labels)
    exc_mask = torch.isin(exc_all_1d, torch.tensor(exc_labels, device=exc_all_1d.device))

    ds = TensorDataset(
    acc_all[exc_mask],
    psd_all[exc_mask],
    exc_all_1d[exc_mask],
    dmg_all_1d[exc_mask],
    )
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    x, x_hat, psd, psd_hat, z_ndmg, z_dmg, exc, dmg= collect_embeddings(data_loader)

    # exc_tr_1d = exc_tr.view(-1)
    exc_mask_tr = torch.isin(exc_tr_1d, torch.tensor(exc_labels))
    # use to get the threshold of Mahalanobis distances from self-supervised training
    ds_tr = TensorDataset(
        acc_tr[exc_mask_tr],
        psd_tr[exc_mask_tr],
        exc_tr_1d[exc_mask_tr],
        dmg_tr_1d[exc_mask_tr],
    )
    data_loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False)
    _, _, _, _, z_ndmg_tr, z_dmg_tr, _, dmg_tr= collect_embeddings(data_loader_tr)


    mahal_dmg, thr_dmg = cds.plot_mahal_baseline_train(z_dmg, z_dmg_tr, exc, dmg, dmg_tr,
                                                            p=percentile, name="z_dmg", 
                                                            enable_plot=False)
    mahal_ndmg, thr_ndmg = cds.plot_mahal_baseline_train(z_ndmg, z_ndmg_tr, exc, dmg, dmg_tr,
                                                            p=percentile, name="z_ndmg", 
                                                            enable_plot=False)

    print("involved excitation labels:", exc_labels)

    cm_dmg = eval_damage_by_percentile_baseline_train(mahal_dmg, thr_dmg, dmg, percentile, healthy_label=1, name="z_dmg mahal")
    cm_ndmg = eval_damage_by_percentile_baseline_train(mahal_ndmg, thr_ndmg, dmg, percentile, healthy_label=1, name="z_ndmg mahal")

    cm_blocks_dmg.append(np.asarray(cm_dmg, dtype=int))
    cm_blocks_ndmg.append(np.asarray(cm_ndmg, dtype=int))

# Concatenate 5 confusion matrices horizontally: shape (2, 10)
cm_dmg_combo = np.hstack(cm_blocks_dmg)
cm_ndmg_combo = np.hstack(cm_blocks_ndmg)

def print_cm_for_excel(cm_combo):
    for row in cm_combo:
        print("\t".join(map(str, row)))
# %%
print(ckpt_name)
print("z_dmg")
print_cm_for_excel(cm_dmg_combo)
# %%
print("z_ndmg")
print_cm_for_excel(cm_ndmg_combo)