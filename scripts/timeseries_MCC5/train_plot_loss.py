# %% 
# load libraries
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn  # add (used by PSD head)

from models.autoencoder import TwoHeadAutoEncoder1D
from models.mlp import PSDHead
from utils.losses import vicreg_loss_individual
from utils.save_model import save_checkpoint
from utils.plot_loss import plot_loss_curves

import time

import matplotlib.pyplot as plt

def log_epoch_losses(
    store: dict,
    split: str,
    n_samples: int,
    total_sum: float,
    l1_sum: float,
    l2a_sum: float,
    l3_sum: float,
):
    """
    Convert per-epoch summed losses (sum over batches weighted by batch size)
    into averages, and append into store dict.

    store structure:
      store["train"]["total"], store["val"]["l1"], ...
    """
    store[split]["total"].append(total_sum / n_samples)
    store[split]["l1"].append(l1_sum / n_samples)
    store[split]["l2a"].append(l2a_sum / n_samples)
    store[split]["l3"].append(l3_sum / n_samples)

from scripts.timeseries_MCC5.configs import (
    dataset_splits_path,
    lr,
    weight_decay,
    epochs,
    batch_size,
    results_dir,
    ckpt_name,
    curve_name,
    h_dim,
    z_dmg_ch,
    z_ndmg_ch,
    lam_time,
    lam_self,
    lam_psd,
    psd_nfft,            # choose multiple of 16 for your conv AE
    psd_to_db,           # True => 10*log10(PSD + eps)
    psd_eps,
    psd_K,
)

import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB__SERVICE_WAIT"] = "15"
os.environ["WANDB_IGNORE_GLOBS"] = "output.log,requirements.txt,wandb-metadata.json,config.yaml"
import wandb

print(
    f"Loss weights | "
    f"lam_time={lam_time}, "
    f"lam_self1={lam_self}, "
    f"lam_psd={lam_psd}"
)
# %% ###########################################################################
# load data

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# results paths
results_path = root / results_dir
ckpt_path = results_path / ckpt_name
curve_path = results_path / curve_name

# load tensors
splits = torch.load(root / dataset_splits_path, map_location="cpu")

acc_tr, psd_tr, exc_tr, dmg_tr = splits["acc_tr"], splits["psd_tr"], splits["exc_tr"].view(-1), splits["dmg_tr"].view(-1)
acc_va, psd_va, exc_va, dmg_va = splits["acc_va"], splits["psd_va"], splits["exc_va"].view(-1), splits["dmg_va"].view(-1)

tr_loader = DataLoader(TensorDataset(acc_tr, psd_tr, exc_tr, dmg_tr), batch_size=batch_size, shuffle=True)
va_loader = DataLoader(TensorDataset(acc_va, psd_va, exc_va, dmg_va), batch_size=batch_size, shuffle=False)

print("train size:", len(exc_tr))
print("validation size:", len(exc_va))

# model
in_ch = acc_tr.shape[1]
num_exc = int(torch.max(exc_tr).item() + 1)

model = TwoHeadAutoEncoder1D(in_ch=in_ch, z_dmg_ch=z_dmg_ch, z_ndmg_ch=z_ndmg_ch).to(device)
psd_head = PSDHead(in_dim=z_dmg_ch, out_ch=in_ch, nfft=psd_nfft, hidden=h_dim).to(device)

opt = AdamW(list(model.parameters()) + list(psd_head.parameters()), lr=lr, weight_decay=weight_decay)

loss_hist = {
    "train": {"total": [], "l1": [], "l2a": [], "l3": []},
    "val":   {"total": [], "l1": [], "l2a": [], "l3": []},
}
# %% ################################################################################################
# training loop
t_start = time.time()
for epoch in range(1, epochs + 1):
    # training
    model.train()
    psd_head.train()

    tr_sum = 0.0
    l1_tr_sum = l2a_tr_sum = l3_tr_sum = 0.0

    for x, psd, exc, dmg in tr_loader:
        x = x.to(device)
        exc = exc.to(device).long()
        dmg = dmg.to(device).long()

        x_hat, z_ndmg, z_dmg = model(x)
        h_dmg  = z_dmg.mean(dim=2)    # (B, z_dmg_ch)
        h_ndmg = z_ndmg.mean(dim=2)   # (B, z_ndmg_ch)

        # loss 1: reconstruction ###########################################
        loss1_tr = F.mse_loss(x_hat, x)

        # loss 3: reconstruct PSD from h_dmg only ###########################################
        psd_hat = psd_head(h_dmg)
        psd_hat = psd_hat[:, :, :psd_K]
        if psd_to_db == True:
            psd_tgt = 10 * torch.log10(psd.to(device) + psd_eps)
            psd_tgt = psd_tgt[:, :, :psd_K]
            psd_tgt = psd_tgt - torch.mean(psd_tgt, dim=(1, 2), keepdim=True)
        else:
            psd_tgt = psd[:, :, :psd_K].to(device)
            psd_tgt = psd_tgt - torch.mean(psd_tgt, dim=(1, 2), keepdim=True)
        loss3_tr = F.mse_loss(psd_hat, psd_tgt)

        # loss 2: apply contrastive learning on data from 5 May ###########################################
        mask = (dmg == 1)
        vicreg_sim_loss_tr = x.new_zeros(())
        vicreg_var_loss_tr = x.new_zeros(())
        vicreg_cov_loss_tr = x.new_zeros(())
        loss2a_tr = x.new_zeros(())
        
        # loss2a, vicreg loss - random pairs ##
        if mask.sum() >= 4:  # need at least 2 pairs for stable var/cov
            hb_dmg = h_dmg[mask]  # (M, D)
            M = hb_dmg.shape[0]
            m2 = (M // 2) * 2
            hb_dmg = hb_dmg[:m2]  # make even
            h1_dmg = hb_dmg[0::2]
            h2_dmg = hb_dmg[1::2]
            vicreg_sim_loss_tr, vicreg_var_loss_tr, vicreg_cov_loss_tr = \
                vicreg_loss_individual(h1_dmg, h2_dmg, var_eps=1e-4)
            loss2a_tr = 25*vicreg_sim_loss_tr + 25*vicreg_var_loss_tr + 1*vicreg_cov_loss_tr

        # total loss
        loss = lam_time * loss1_tr + \
                lam_self * loss2a_tr + \
                lam_psd * loss3_tr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = x.size(0)
        l1_tr_sum += loss1_tr.item() * bs
        l2a_tr_sum += loss2a_tr.item() * bs
        l3_tr_sum  += loss3_tr.item() * bs

        tr_sum += loss.item() * bs

    log_epoch_losses(
        loss_hist, "train", len(exc_tr),
        total_sum=tr_sum,
        l1_sum=l1_tr_sum,
        l2a_sum=l2a_tr_sum,
        l3_sum=l3_tr_sum,
        )
    tr_loss = loss_hist["train"]["total"][-1]

    # validation
    model.eval()
    psd_head.eval()

    va_sum = 0.0
    l1_va_sum = l2a_va_sum = l3_va_sum = 0.0
    with torch.no_grad():
        for x, psd, exc, dmg in va_loader:
            x = x.to(device)
            exc = exc.to(device).long()
            dmg = dmg.to(device).long()

            x_hat, z_ndmg, z_dmg= model(x)
            h_dmg  = z_dmg.mean(dim=2)    # (B, z_dmg_ch)
            h_ndmg = z_ndmg.mean(dim=2)   # (B, z_ndmg_ch)

            # loss 1: reconstruction
            loss1_va = F.mse_loss(x_hat, x)

            # loss 3: reconstruct normalized log-PSD from h_dmg only ###########################################
            psd_hat = psd_head(h_dmg)
            psd_hat = psd_hat[:, :, :psd_K]
            if psd_to_db == True:
                psd_tgt = 10 * torch.log10(psd.to(device) + psd_eps)
                psd_tgt = psd_tgt[:, :, :psd_K]
                psd_tgt = psd_tgt - torch.mean(psd_tgt, dim=(1, 2), keepdim=True)
            else:
                psd_tgt = psd[:, :, :psd_K].to(device)
                psd_tgt = psd_tgt - torch.mean(psd_tgt, dim=(1, 2), keepdim=True)
            loss3_va = F.mse_loss(psd_hat, psd_tgt)

            # loss 2: apply contrastive learning on data from 5 May
            mask = (dmg == 1)
            vicreg_sim_loss_va = x.new_zeros(())
            vicreg_var_loss_va = x.new_zeros(())
            vicreg_cov_loss_va = x.new_zeros(())
            loss2a_va = x.new_zeros(())

            # loss 2a: vicreg loss using random pairs #########
            if mask.sum() >= 4:  # need at least 2 pairs for stable var/cov
                hb_dmg = h_dmg[mask]  # (M, D)
                M = hb_dmg.shape[0]
                m2 = (M // 2) * 2
                hb_dmg = hb_dmg[:m2]  # make even
                h1_dmg = hb_dmg[0::2]
                h2_dmg = hb_dmg[1::2]
                vicreg_sim_loss_va, vicreg_var_loss_va, vicreg_cov_loss_va = \
                    vicreg_loss_individual(h1_dmg, h2_dmg, var_eps=1e-4)
                loss2a_va = 25*vicreg_sim_loss_va + 25*vicreg_var_loss_va + 1*vicreg_cov_loss_va

            # total loss
            loss = lam_time * loss1_va + \
                lam_self * loss2a_va + \
                lam_psd * loss3_va
            
            bs = x.size(0)
            l1_va_sum += loss1_va.item() * bs
            l2a_va_sum += loss2a_va.item() * bs
            l3_va_sum  += loss3_va.item() * bs

            va_sum += loss.item() * bs

    log_epoch_losses(
        loss_hist, "val", len(exc_va),
        total_sum=va_sum,
        l1_sum=l1_va_sum,
        l2a_sum=l2a_va_sum,
        l3_sum=l3_va_sum,
    )
    va_loss = loss_hist["val"]["total"][-1]

    print(f"epoch {epoch}  train {tr_loss:.6f}  val {va_loss:.6f}")

t_total = time.time() - t_start
print(f"[TIME] Total training time: {t_total:.2f} seconds ({t_total/60:.2f} minutes)")
# %% 
# save and plot loss

import matplotlib as mpl
def plot_train_val_losses(loss_hist: dict, save_path=None, title="Loss curves"):
    """
    Plot train/val curves for total, l1, l2a, l3.
    One figure with 4 panels for readability.
    """
    # ===== Global font settings =====
    mpl.rcParams.update({"font.size": 10.5})

    epochs = np.arange(1, len(loss_hist["train"]["total"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
    axes = axes.ravel()

    panels = [
        ("total", "Total loss"),
        ("l1",    "Loss 1: time recon"),
        ("l3",    "Loss 2: PSD recon"),
        ("l2a",   "Loss 3: VICReg"),
    ]

    for ax, (k, ttl) in zip(axes, panels):
        ax.plot(epochs, loss_hist["train"][k], label=f"train")
        ax.plot(epochs, loss_hist["val"][k],   label=f"validation")
        ax.set_title(ttl)
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-10, 510) 
        ax.set_xticks(np.arange(0, 501, 100))

    axes[2].set_xlabel("Epoch")
    axes[3].set_xlabel("Epoch")
    # fig.suptitle(title)
    fig.tight_layout()

    # if save_path is not None:
    #     fig.savefig(save_path, dpi=200)
    plt.show()

# save plot
plot_train_val_losses(
    loss_hist,
    save_path=str(curve_path),
    title=f"Loss curves | {ckpt_name}"
)

# optionally also dump raw numbers for later aggregation
# np.savez(
#     str(curve_path).replace(".png", ".npz"),
#     train_total=np.array(loss_hist["train"]["total"]),
#     train_l1=np.array(loss_hist["train"]["l1"]),
#     train_l2a=np.array(loss_hist["train"]["l2a"]),
#     train_l3=np.array(loss_hist["train"]["l3"]),
#     val_total=np.array(loss_hist["val"]["total"]),
#     val_l1=np.array(loss_hist["val"]["l1"]),
#     val_l2a=np.array(loss_hist["val"]["l2a"]),
#     val_l3=np.array(loss_hist["val"]["l3"]),
# )
# print("Saved loss curves to:", curve_path)
# print("Saved loss arrays to :", str(curve_path).replace(".png", ".npz"))