# paths #######################################
mat_path = "/cluster/work/ibk_chatzi/Xudong/CFMD/MCC5/data_time.mat" # ETH EULER

results_dir = "results/timeseries_MCC5"

# dataset generation #######################################
field_exc = "excitation_label"
field_dmg = "damage_label"

seed = 999
split = (0.8, 0.2, 0.0)


# v1 #######################################
fs = 3200

window_s = 4096 / fs
stride_s = 4096 / 2 / fs  # stride_s = 0, interpreted as no overlap

# PSD (Welch) parameters
psd_nperseg = 1024
psd_noverlap = 512
psd_nfft = 2048            # choose multiple of 16 for your conv AE
psd_to_db = True           # True => 10*log10(PSD + eps)
psd_eps = 1e-12
psd_K = 1024+1   # the first K points of psds

# use raw time series
dataset_splits_path = "/cluster/work/ibk_chatzi/Xudong/CFMD/MCC5/dataset_splits_timeseries_psd.pt" # ETH EULER
field_acc = "acc" # raw acc
field_load = "load" # raw acc