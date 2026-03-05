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

from scripts.timeseries_MCC5.configs import (
    dataset_splits_path,
    lr,
    weight_decay,
    epochs,
    batch_size,
    results_dir,
    ckpt_name,
    curve_name,
    window_s,
    stride_s,
    h_dim,
    z_dmg_ch,
    z_ndmg_ch,
    lam_time,
    lam_self,
    lam_psd,
    psd_nperseg,
    psd_noverlap,
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
    f"lam_self={lam_self}, "
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
splits = torch.load(root/dataset_splits_path, map_location="cpu")

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

best_va = 1e30
tr_losses, va_losses = [], []
# %% ################################################################################################
# training loop
t_start = time.time()
for epoch in range(1, epochs + 1):
    # training
    model.train()
    psd_head.train()

    tr_sum = 0.0
    l1_tr_sum = l2_tr_sum = l5_tr_sum = 0.0
    sim_tr_sum = var_tr_sum = cov_tr_sum = 0.0

    for x, psd, exc, dmg in tr_loader:
        x = x.to(device)
        exc = exc.to(device).long()
        dmg = dmg.to(device).long()

        x_hat, z_ndmg, z_dmg = model(x)
        h_dmg  = z_dmg.mean(dim=2)    # (B, z_dmg_ch)
        h_ndmg = z_ndmg.mean(dim=2)   # (B, z_ndmg_ch)

        # loss 1: reconstruction ###########################################
        loss1_tr = F.mse_loss(x_hat, x)

        # loss 5: reconstruct PSD from h_dmg only ###########################################
        psd_hat = psd_head(h_dmg)
        psd_hat = psd_hat[:, :, :psd_K]
        if psd_to_db == True:
            psd_tgt = 10 * torch.log10(psd.to(device) + psd_eps)
            psd_tgt = psd_tgt[:, :, :psd_K]
            psd_tgt = psd_tgt - torch.mean(psd_tgt, dim=(1, 2), keepdim=True)
        else:
            psd_tgt = psd[:, :, :psd_K].to(device)
            psd_tgt = psd_tgt - torch.mean(psd_tgt, dim=(1, 2), keepdim=True)
        loss5_tr = F.mse_loss(psd_hat, psd_tgt)

        # loss 2: apply contrastive learning on data from dmg label 1 ###########################################
        mask = (dmg == 1)
        vicreg_sim_loss_tr = x.new_zeros(())
        vicreg_var_loss_tr = x.new_zeros(())
        vicreg_cov_loss_tr = x.new_zeros(())
        loss2_tr = x.new_zeros(())
        
        # loss 2a: vicreg loss
        # vicreg loss - random pairs ##
        if mask.sum() >= 4:  # need at least 2 pairs for stable var/cov
            hb_dmg = h_dmg[mask]  # (M, D)
            M = hb_dmg.shape[0]
            m2 = (M // 2) * 2
            hb_dmg = hb_dmg[:m2]  # make even
            h1_dmg = hb_dmg[0::2]
            h2_dmg = hb_dmg[1::2]
            vicreg_sim_loss_tr, vicreg_var_loss_tr, vicreg_cov_loss_tr = \
                vicreg_loss_individual(h1_dmg, h2_dmg, var_eps=1e-4)
            loss2_tr = 25*vicreg_sim_loss_tr + 25*vicreg_var_loss_tr + 1*vicreg_cov_loss_tr

        # total loss
        loss = lam_time * loss1_tr + \
                lam_self * loss2_tr + \
                lam_psd * loss5_tr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = x.size(0)
        l1_tr_sum += loss1_tr.item() * bs
        l2_tr_sum += loss2_tr.item() * bs
        l5_tr_sum  += loss5_tr.item() * bs

        sim_tr_sum += vicreg_sim_loss_tr.item() * bs
        var_tr_sum += vicreg_var_loss_tr.item() * bs
        cov_tr_sum += vicreg_cov_loss_tr.item() * bs

        tr_sum += loss.item() * bs

    tr_loss = tr_sum / len(exc_tr)
    tr_losses.append(tr_loss)

    # validation
    model.eval()
    psd_head.eval()

    va_sum = 0.0
    l1_va_sum = l2_va_sum = l5_va_sum = 0.0
    sim_va_sum = var_va_sum = cov_va_sum = 0.0
    with torch.no_grad():
        for x, psd, exc, dmg in va_loader:
            x = x.to(device)
            dmg = dmg.to(device).long()
            exc = exc.to(device).long()

            x_hat, z_ndmg, z_dmg= model(x)
            h_dmg  = z_dmg.mean(dim=2)    # (B, z_dmg_ch)
            h_ndmg = z_ndmg.mean(dim=2)   # (B, z_ndmg_ch)

            # loss 1: reconstruction
            loss1_va = F.mse_loss(x_hat, x)

            # loss 5: reconstruct normalized log-PSD from h_dmg only ###########################################
            psd_hat = psd_head(h_dmg)
            psd_hat = psd_hat[:, :, :psd_K]
            if psd_to_db == True:
                psd_tgt = 10 * torch.log10(psd.to(device) + psd_eps)
                psd_tgt = psd_tgt[:, :, :psd_K]
                psd_tgt = psd_tgt - torch.mean(psd_tgt, dim=(1, 2), keepdim=True)
            else:
                psd_tgt = psd[:, :, :psd_K].to(device)
                psd_tgt = psd_tgt - torch.mean(psd_tgt, dim=(1, 2), keepdim=True)
            loss5_va = F.mse_loss(psd_hat, psd_tgt)

            # loss 2: apply contrastive learning on data from 5 May
            mask = (dmg == 1)
            vicreg_sim_loss_va = x.new_zeros(())
            vicreg_var_loss_va = x.new_zeros(())
            vicreg_cov_loss_va = x.new_zeros(())
            loss2_va = x.new_zeros(())

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
                loss2_va = 25*vicreg_sim_loss_va + 25*vicreg_var_loss_va + 1*vicreg_cov_loss_va

            # total loss
            loss = lam_time * loss1_va + \
                    lam_self * loss2_va + \
                    lam_psd * loss5_va
            
            bs = x.size(0)
            l1_va_sum += loss1_va.item() * bs
            l2_va_sum += loss2_va.item() * bs
            l5_va_sum  += loss5_va.item() * bs

            sim_va_sum += vicreg_sim_loss_va.item() * bs
            var_va_sum += vicreg_var_loss_va.item() * bs
            cov_va_sum += vicreg_cov_loss_va.item() * bs

            va_sum += loss.item() * bs

    va_loss = va_sum / len(exc_va)
    va_losses.append(va_loss)

    print(f"epoch {epoch}  train {tr_loss:.6f}  val {va_loss:.6f}")

    # save model
    # if va_loss < best_va:
    #     best_va = va_loss
    #     best_epoch = epoch
    #     save_checkpoint(str(ckpt_path), model, epoch, best_va)

    #     torch.save(
    #     {
    #         "psd_head": psd_head.state_dict(),
    #     },
    #     str(ckpt_path).replace(".pt", "_psd_head.pt"),
    #     )
t_total = time.time() - t_start
print(f"[TIME] Total training time: {t_total:.2f} seconds ({t_total/60:.2f} minutes)")
# %% 
# save loss history
# np.savez(str(log_path), tr_losses=np.array(tr_losses), va_losses=np.array(va_losses))
plot_loss_curves(tr_losses, va_losses, str(curve_path))

# save model
best_va = va_loss
best_epoch = epoch
save_checkpoint(str(ckpt_path), model, epoch, best_va)
torch.save(
{
    "psd_head": psd_head.state_dict(),
},
str(ckpt_path).replace(".pt", "_psd_head.pt"),
)

print("saved checkpoint:", ckpt_path)
print("saved curve:", curve_path)
print("best epoch:", best_epoch)
print("best validation loss:", best_va)