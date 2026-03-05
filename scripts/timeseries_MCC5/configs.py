# paths #######################################
dataset_splits_path = "data/MCC5.pt" # ETH EULER
results_dir = "results/timeseries_MCC5"

# dataset generation #######################################
field_exc = "excitation_label"
field_dmg = "damage_label"
field_acc = "acc" # raw acc
field_load = "load" # raw acc

seed = 999
split = (0.6, 0.2, 0.2)

fs = 3200

window_s = 2048 / fs
stride_s = window_s / 2  # stride_s = 0, interpreted as no overlap

# PSD (Welch) parameters
psd_nperseg = 1024
psd_noverlap = 512
psd_nfft = 2048            # choose multiple of 16 for conv AE
psd_to_db = True           # True => 10*log10(PSD + eps)
psd_eps = 1e-12
psd_K = 1024+1   # the first K points of psds

# model hyper-parameters #######################################
# two head AE
z_dmg_ch = 128
z_ndmg_ch = 128
# MLP
h_dim = 512

# training settings, ######################################
batch_size = 256
lr = 1e-3
weight_decay = 1e-4
epochs = 500
# loss coefficients
lam_time = 100
lam_self = 1
lam_psd = 100

# time recon +VICReg + PSD   100,1,100
ckpt_name = "model_o_4.pt"  
curve_name = "loss_curves_o_4.png"