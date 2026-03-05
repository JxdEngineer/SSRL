import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import datetime, timezone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from typing import Iterable, Optional, Sequence, Tuple, Union

ArrayLike = Union[np.ndarray, "object"]  # torch.Tensor also works via np.asarray

def _to_numpy(x):
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _reduce_latent(Z: ArrayLike) -> np.ndarray:
    """
    Z can be:
      (N, C, Lz) time latent, will be reduced to (N, C) by mean over axis=2
      (N, D) already reduced
    """
    Z = _to_numpy(Z)
    if Z.ndim == 3:
        return Z.mean(axis=2)
    if Z.ndim == 2:
        return Z
    raise ValueError(f"Expected Z with 2 or 3 dims, got shape {Z.shape}")

def plot_mahal_vs_time(  # for OpenLab dataset
    z, dt, exc, dmg, p,
    name="z",
    baseline_date_utc=datetime(2025, 5, 5, 0, 0, 0, tzinfo=timezone.utc),
    per_exc_baseline=False,
    enable_plot=True,
):
    """
    z: (N, C, Lz) or (N, C*Lz)
    dt: (N,) unix seconds
    exc: (N,) excitation labels (int)
    per_exc_baseline: if True, fit baseline Gaussian per excitation and score with matched model
    """
    
    h = _reduce_latent(z)
    dt = _to_numpy(dt).reshape(-1)
    exc = _to_numpy(exc).reshape(-1)
    dmg = _to_numpy(dmg).reshape(-1)

    # baseline mask: baseline_date_utc day only (UTC)
    sec_per_day = 24 * 3600
    d0 = baseline_date_utc.timestamp()
    day_idx = np.floor(((dt - d0) // sec_per_day)).astype(int)
    baseline_mask = (day_idx == 0)

    print(f"[{name}] baseline count (5 May):", int(baseline_mask.sum()), "/", h.shape[0])
    if baseline_mask.sum() < 2:
        raise ValueError(f"[{name}] baseline has too few samples: {int(baseline_mask.sum())}")

    mahal = np.full(h.shape[0], np.nan, dtype=np.float64)

    if not per_exc_baseline:
        Hb = h[baseline_mask]
        mu = Hb.mean(axis=0, keepdims=True)
        lw = LedoitWolf().fit(Hb)
        Sigma_inv = lw.precision_
        diff = h - mu
        mahal = np.einsum("nd,dd,nd->n", diff, Sigma_inv, diff)
    else:
        # fit and score per excitation
        uniq_exc = np.unique(exc)
        for e in uniq_exc:
            m_all = (exc == e)
            m_base = baseline_mask & m_all
            Hb_e = h[m_base]

            # fallback if too few baseline samples for this excitation
            if Hb_e.shape[0] < 5:
                # use global baseline for this excitation group
                Hb = h[baseline_mask]
                mu = Hb.mean(axis=0, keepdims=True)
                lw = LedoitWolf().fit(Hb)
                Sigma_inv = lw.precision_
                diff = h[m_all] - mu
                mahal[m_all] = np.einsum("nd,dd,nd->n", diff, Sigma_inv, diff)
                continue

            mu_e = Hb_e.mean(axis=0, keepdims=True)
            lw_e = LedoitWolf().fit(Hb_e)
            Sig_inv_e = lw_e.precision_
            diff_e = h[m_all] - mu_e
            mahal[m_all] = np.einsum("nd,dd,nd->n", diff_e, Sig_inv_e, diff_e)

    # baseline stats in Mahalanobis space (using whichever scoring mode)
    mahal_baseline = mahal[baseline_mask]
    baseline_mean = float(np.mean(mahal_baseline))
    baseline_median = float(np.median(mahal_baseline))
    baseline_p = float(np.percentile(mahal_baseline, p))

    # sort by datetime for plotting
    dt_readable = np.array([datetime.fromtimestamp(float(t), tz=timezone.utc) for t in dt])
    order = np.argsort(dt_readable)
    dt_sorted = dt_readable[order]
    exc_sorted = exc[order]
    dmg_sorted = dmg[order]

    # plot mahal
    if enable_plot:
        plt.figure(figsize=(10, 6))
        if np.size(np.unique(exc_sorted)) > 1:
            for e in np.unique(exc_sorted):
                idx = (exc_sorted == e)
                plt.scatter(dt_sorted[idx], mahal[order][idx], s=10, label=f"exc {int(e)}")
        else:
            for d in np.unique(dmg_sorted):
                idx = (dmg_sorted == d)
                plt.scatter(dt_sorted[idx], mahal[order][idx], s=10, label=f"dmg {int(d)}")

        plt.axhline(baseline_mean, linestyle="--", linewidth=1.5, label=f"baseline mean ({baseline_mean:.2g})")
        plt.axhline(baseline_median, linestyle=":", linewidth=1.8, label=f"baseline median ({baseline_median:.2g})")
        plt.axhline(baseline_p, linestyle="-.", linewidth=1.5, label=f"baseline p{p} ({baseline_p:.2g})")

        plt.xlabel("Datetime (UTC)")
        ylabel_suffix = "per exc baseline" if per_exc_baseline else "global baseline"
        plt.ylabel(f"Mahalanobis distance ({ylabel_suffix})")
        plt.yscale("log")

        plt.title(f"Unsupervised score from {name} (Mahalanobis)")
        plt.xticks(rotation=30, ha="right")
        plt.legend(markerscale=1.5, fontsize=9, loc='best')
        plt.tight_layout()
        plt.show()

    return mahal, baseline_mask


def plot_mahal_vs_time_baseline_train(  # for OpenLab dataset
    z, dt, z_tr, dt_tr, exc, dmg, p,
    name="z",
    baseline_date_utc=datetime(2025, 5, 5, 0, 0, 0, tzinfo=timezone.utc),
    enable_plot=True,
):
    """
    z: (N, C, Lz) or (N, C*Lz)
    dt: (N,) unix seconds
    exc: (N,) excitation labels (int)
    per_exc_baseline: if True, fit baseline Gaussian per excitation and score with matched model
    """
    
    h = _reduce_latent(z)
    h_tr = _reduce_latent(z_tr)
    dt_tr = _to_numpy(dt_tr).reshape(-1)
    exc = _to_numpy(exc).reshape(-1)
    dmg = _to_numpy(dmg).reshape(-1)

    # baseline mask: baseline_date_utc day only (UTC)
    sec_per_day = 24 * 3600
    d0 = baseline_date_utc.timestamp()
    day_idx = np.floor(((dt_tr - d0) // sec_per_day)).astype(int)
    baseline_mask = (day_idx == 0)

    mahal = np.full(h.shape[0], np.nan, dtype=np.float64)

    Hb_tr = h_tr[baseline_mask]
    mu_tr = Hb_tr.mean(axis=0, keepdims=True)
    lw_tr = LedoitWolf().fit(Hb_tr)
    Sigma_inv_tr = lw_tr.precision_

    diff = h - mu_tr
    mahal = np.einsum("nd,dd,nd->n", diff, Sigma_inv_tr, diff)

    diff_tr = h_tr - mu_tr
    mahal_tr = np.einsum("nd,dd,nd->n", diff_tr, Sigma_inv_tr, diff_tr)

    # baseline stats in Mahalanobis space (using whichever scoring mode)
    mahal_baseline = mahal_tr[baseline_mask]
    baseline_mean = float(np.mean(mahal_baseline))
    baseline_median = float(np.median(mahal_baseline))
    baseline_p = float(np.percentile(mahal_baseline, p))

    # sort by datetime for plotting
    dt_readable = np.array([datetime.fromtimestamp(float(t), tz=timezone.utc) for t in dt])
    order = np.argsort(dt_readable)
    dt_sorted = dt_readable[order]
    exc_sorted = exc[order]
    dmg_sorted = dmg[order]

    # plot mahal
    if enable_plot:
        # ===== Global font settings =====
        mpl.rcParams.update({"font.size": 10.5})

        # color by excitation
        if np.size(np.unique(exc_sorted)) > 1:
            plt.figure(figsize=(10, 6))
            for e in np.unique(exc_sorted):
                idx = (exc_sorted == e)
                plt.scatter(dt_sorted[idx], mahal[order][idx], s=10, label=f"exc {int(e)}")

            plt.axhline(baseline_mean, linestyle="--", linewidth=1.5, label=f"baseline mean ({baseline_mean:.2g})")
            plt.axhline(baseline_median, linestyle=":", linewidth=1.8, label=f"baseline median ({baseline_median:.2g})")
            plt.axhline(baseline_p, linestyle="-.", linewidth=1.5, label=f"baseline p{p} ({baseline_p:.2g})")

            plt.xlabel("Datetime (CET)")
            plt.ylabel(f"Mahalanobis distance")
            plt.yscale("log")

            plt.title(f"Unsupervised score from {name} (Mahalanobis)")
            plt.xticks(rotation=30, ha="right")
            plt.legend(markerscale=1.5, fontsize=9, loc='best')
            plt.tight_layout()
            plt.show()

        # color by damage
        plt.figure(figsize=(6, 6))
        for d in np.unique(dmg_sorted):
            idx = (dmg_sorted == d)
            plt.scatter(dt_sorted[idx], mahal[order][idx], s=6, label=f"dmg {int(d)}")

        # plt.axhline(baseline_mean, linestyle="--", linewidth=2, label=f"baseline mean ({baseline_mean:.2g})")
        plt.axhline(baseline_median, linestyle=":", linewidth=2, label=f"baseline p50 ({baseline_median:.2g})")
        plt.axhline(baseline_p, linestyle="-", linewidth=2, label=f"baseline p{p} ({baseline_p:.2g})")

        # plt.axhline(baseline_mean, linestyle="--", linewidth=1.5, label=f"baseline mean")
        # plt.axhline(baseline_median, linestyle=":", linewidth=1.8, label=f"baseline median")
        # plt.axhline(baseline_p, linestyle="-.", linewidth=1.5, label=f"baseline p{p}")

        plt.xlabel("Datetime (CET)")
        plt.ylabel(f"Mahalanobis distance ({name})")
        plt.yscale("log")

        # X ticks every 3 hours
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        # Grid
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(rotation=45, ha="right")

        # Legend inside with 3 rows
        handles, labels = plt.gca().get_legend_handles_labels()
        n_items = len(labels)
        ncol = int(np.ceil(n_items / 4))
        plt.legend(handles, labels,
           ncol=ncol,
           markerscale=1.5,
           fontsize=9,
           loc='lower right')
        
        plt.tight_layout()
        plt.show()

    return mahal, baseline_p




def plot_mahal(
    z: ArrayLike,
    exc: ArrayLike,
    dmg: ArrayLike,
    p: float = 95,
    name: str = "z",
    healthy_label: int = 1,
    per_exc_baseline: bool = False,
    enable_plot: bool = True,
    start_dt_utc: datetime = datetime(2026, 2, 8, 0, 0, 0, tzinfo=timezone.utc),
    step_seconds: int = 60,
):
    """
    Dataset without dt: baseline by dmg==healthy_label; synth dt by (damage, excitation) grid.

    Synthetic dt assignment rule (your request):
      For each damage label d, excitations are laid out in time order:
        dmg d exc 1 -> t0 + offset minutes
        dmg d exc 2 -> t0 + (offset+1) minutes
        dmg d exc 3 -> t0 + (offset+2) minutes
      Then next damage label continues:
        dmg (d+1) exc 1 -> t0 + (offset+E) minutes
      where E = number of unique excitation labels (sorted).

    Returns
    mahal: (N,) squared Mahalanobis distances
    baseline_mask: (N,) bool (dmg==healthy_label)
    dt: (N,) unix seconds (synthetic)
    """
    h = _reduce_latent(z)  # (N, D)
    dmg = _to_numpy(dmg).reshape(-1)
    exc = _to_numpy(exc).reshape(-1)

    if h.shape[0] != dmg.shape[0] or h.shape[0] != exc.shape[0]:
        raise ValueError(f"[{name}] length mismatch: z {h.shape[0]}, dmg {dmg.shape[0]}, exc {exc.shape[0]}")

    # baseline by healthy label
    baseline_mask = (dmg == healthy_label)
    print(f"[{name}] baseline count (dmg=={healthy_label}): {int(baseline_mask.sum())} / {h.shape[0]}")
    if baseline_mask.sum() < 2:
        raise ValueError(f"[{name}] baseline has too few samples: {int(baseline_mask.sum())}")

    # --- synthetic dt by (damage, excitation) ---
    # Map excitation labels to contiguous indices 0..E-1 based on sorted unique labels
    uniq_exc = np.unique(exc)
    uniq_exc_sorted = np.sort(uniq_exc.astype(int))
    E = len(uniq_exc_sorted)
    if E < 1:
        raise ValueError(f"[{name}] no excitation labels found")

    exc_to_idx = {int(e): i for i, e in enumerate(uniq_exc_sorted)}
    exc_idx = np.array([exc_to_idx[int(e)] for e in exc], dtype=np.int64)  # 0..E-1

    dmg_min = int(np.min(dmg))
    if dmg_min < 1:
        raise ValueError(f"[{name}] expected dmg labels start at 1, got min dmg={dmg_min}")

    base_ts = int(start_dt_utc.timestamp())

    # time slot index = (dmg-1)*E + exc_idx
    slot = (dmg.astype(np.int64) - 1) * int(E) + exc_idx
    dt = base_ts + slot * int(step_seconds)

    # --- Mahalanobis scoring ---
    mahal = np.full(h.shape[0], np.nan, dtype=np.float64)

    if not per_exc_baseline:
        Hb = h[baseline_mask]
        mu = Hb.mean(axis=0, keepdims=True)
        lw = LedoitWolf().fit(Hb)
        Sig_inv = lw.precision_
        diff = h - mu
        mahal = np.einsum("nd,dd,nd->n", diff, Sig_inv, diff)
    else:
        # Fit baseline Gaussian per excitation (using healthy samples within each excitation),
        # fallback to global baseline if too few.
        Hb_global = h[baseline_mask]
        mu_g = Hb_global.mean(axis=0, keepdims=True)
        lw_g = LedoitWolf().fit(Hb_global)
        Sig_inv_g = lw_g.precision_

        for e in uniq_exc_sorted:
            m_all = (exc.astype(int) == int(e))
            m_base = baseline_mask & m_all
            Hb_e = h[m_base]

            if Hb_e.shape[0] < 5:
                diff = h[m_all] - mu_g
                mahal[m_all] = np.einsum("nd,dd,nd->n", diff, Sig_inv_g, diff)
                continue

            mu_e = Hb_e.mean(axis=0, keepdims=True)
            lw_e = LedoitWolf().fit(Hb_e)
            Sig_inv_e = lw_e.precision_
            diff_e = h[m_all] - mu_e
            mahal[m_all] = np.einsum("nd,dd,nd->n", diff_e, Sig_inv_e, diff_e)

    # baseline stats
    mahal_b = mahal[baseline_mask]
    baseline_mean = float(np.mean(mahal_b))
    baseline_median = float(np.median(mahal_b))
    baseline_p = float(np.percentile(mahal_b, p))

    # sort by synthetic time
    order = np.argsort(dt)
    dt_sorted = np.array([datetime.fromtimestamp(float(t), tz=timezone.utc) for t in dt[order]])
    exc_sorted = exc[order]
    dmg_sorted = dmg[order]

    # plot mahal
    if enable_plot:
        plt.figure(figsize=(10, 6))
        if np.size(np.unique(exc_sorted)) > 1:
            for e in np.unique(exc_sorted):
                idx = (exc_sorted == e)
                plt.scatter(dt_sorted[idx], mahal[order][idx], s=10, label=f"exc {int(e)}")
        else:
            for d in np.unique(dmg_sorted):
                idx = (dmg_sorted == d)
                plt.scatter(dt_sorted[idx], mahal[order][idx], s=10, label=f"dmg {int(d)}")

        plt.axhline(baseline_mean, linestyle="--", linewidth=1.5, label=f"baseline mean ({baseline_mean:.2g})")
        plt.axhline(baseline_median, linestyle=":", linewidth=1.8, label=f"baseline median ({baseline_median:.2g})")
        plt.axhline(baseline_p, linestyle="-.", linewidth=1.5, label=f"baseline p{p} ({baseline_p:.2g})")

        plt.xlabel("Synthetic Datetime (UTC)")
        ylabel_suffix = "per exc baseline" if per_exc_baseline else "global baseline"
        plt.ylabel(f"Mahalanobis distance ({ylabel_suffix})")
        plt.yscale("log")

        plt.title(f"Unsupervised score from {name} (Mahalanobis)")
        plt.xticks(rotation=30, ha="right")
        plt.legend(markerscale=1.5, fontsize=9, loc='best')
        plt.tight_layout()
        plt.show()

    return mahal, baseline_mask, dt


def plot_mahal_baseline_train(
    z: ArrayLike,
    z_tr: ArrayLike,
    exc: ArrayLike,
    dmg: ArrayLike,
    dmg_tr: ArrayLike,
    p: float = 95,
    name: str = "z",
    healthy_label: int = 1,
    enable_plot: bool = True,
    start_dt_utc: datetime = datetime(2026, 2, 8, 0, 0, 0, tzinfo=timezone.utc),
    step_seconds: int = 60,
):
    """
    Dataset without dt: baseline by dmg==healthy_label; synth dt by (damage, excitation) grid.

    Synthetic dt assignment rule (your request):
      For each damage label d, excitations are laid out in time order:
        dmg d exc 1 -> t0 + offset minutes
        dmg d exc 2 -> t0 + (offset+1) minutes
        dmg d exc 3 -> t0 + (offset+2) minutes
      Then next damage label continues:
        dmg (d+1) exc 1 -> t0 + (offset+E) minutes
      where E = number of unique excitation labels (sorted).

    Returns
    mahal: (N,) squared Mahalanobis distances
    baseline_mask: (N,) bool (dmg==healthy_label)
    dt: (N,) unix seconds (synthetic)
    """
    h = _reduce_latent(z)  # (N, D)
    h_tr = _reduce_latent(z_tr)  # (N, D)
    dmg = _to_numpy(dmg).reshape(-1)
    dmg_tr = _to_numpy(dmg_tr).reshape(-1)
    exc = _to_numpy(exc).reshape(-1)

     # baseline by healthy label
    baseline_mask = (dmg_tr == healthy_label)
    print(f"[{name}] baseline count (dmg_tr=={healthy_label}): {int(baseline_mask.sum())} / {h_tr.shape[0]}")

    # --- synthetic dt by (damage, excitation) ---
    # Map excitation labels to contiguous indices 0..E-1 based on sorted unique labels
    uniq_exc = np.unique(exc)
    uniq_exc_sorted = np.sort(uniq_exc.astype(int))
    E = len(uniq_exc_sorted)
    if E < 1:
        raise ValueError(f"[{name}] no excitation labels found")

    exc_to_idx = {int(e): i for i, e in enumerate(uniq_exc_sorted)}
    exc_idx = np.array([exc_to_idx[int(e)] for e in exc], dtype=np.int64)  # 0..E-1

    dmg_min = int(np.min(dmg))
    if dmg_min < 1:
        raise ValueError(f"[{name}] expected dmg labels start at 1, got min dmg={dmg_min}")

    base_ts = int(start_dt_utc.timestamp())

    # time slot index = (dmg-1)*E + exc_idx
    slot = (dmg.astype(np.int64) - 1) * int(E) + exc_idx
    dt = base_ts + slot * int(step_seconds)

    # --- Mahalanobis scoring ---
    mahal = np.full(h.shape[0], np.nan, dtype=np.float64)
    mahal_base = np.full(baseline_mask.shape[0], np.nan, dtype=np.float64)

    H_base = h_tr[baseline_mask]
    mu_base = H_base.mean(axis=0, keepdims=True)
    lw_base = LedoitWolf().fit(H_base)
    Sig_inv_base = lw_base.precision_

    diff_base = H_base - mu_base
    mahal_base = np.einsum("nd,dd,nd->n", diff_base, Sig_inv_base, diff_base)

    diff = h - mu_base
    mahal = np.einsum("nd,dd,nd->n", diff, Sig_inv_base, diff)

    # baseline stats
    baseline_mean = float(np.mean(mahal_base))
    baseline_median = float(np.median(mahal_base))
    baseline_p = float(np.percentile(mahal_base, p))

    # sort by synthetic time
    order = np.argsort(dt)
    dt_sorted = np.array([datetime.fromtimestamp(float(t), tz=timezone.utc) for t in dt[order]])
    exc_sorted = exc[order]
    dmg_sorted = dmg[order]

    # plot mahal
    if enable_plot:
        # ===== Global font settings =====
        mpl.rcParams.update({"font.size": 10.5})

        # color by excitation
        if np.size(np.unique(exc_sorted)) > 1:
            plt.figure(figsize=(10, 6))
            for e in np.unique(exc_sorted):
                idx = (exc_sorted == e)
                plt.scatter(dt_sorted[idx], mahal[order][idx], s=10, label=f"exc {int(e)}")

            plt.axhline(baseline_mean, linestyle="--", linewidth=1.5, label=f"baseline mean ({baseline_mean:.2g})")
            plt.axhline(baseline_median, linestyle=":", linewidth=1.8, label=f"baseline median ({baseline_median:.2g})")
            plt.axhline(baseline_p, linestyle="-.", linewidth=1.5, label=f"baseline p{p} ({baseline_p:.2g})")

            plt.xlabel("Synthetic Datetime (CET)")
            plt.ylabel(f"Mahalanobis distance")
            plt.yscale("log")

            plt.title(f"Unsupervised score from {name} (Mahalanobis)")
            plt.xticks(rotation=30, ha="right")
            plt.legend(markerscale=1.5, fontsize=9, loc='best')
            plt.tight_layout()
            plt.show()

        # color by damage
        plt.figure(figsize=(6, 6))
        for d in np.unique(dmg_sorted):
            idx = (dmg_sorted == d)
            plt.scatter(dt_sorted[idx], mahal[order][idx], s=6, label=f"dmg {int(d)}")

        # plt.axhline(baseline_mean, linestyle="--", linewidth=2, label=f"baseline mean ({baseline_mean:.2g})")
        plt.axhline(baseline_median, linestyle=":", linewidth=2, label=f"baseline p50 ({baseline_median:.2g})")
        plt.axhline(baseline_p, linestyle="-", linewidth=2, label=f"baseline p{p} ({baseline_p:.2g})")

        # plt.axhline(baseline_mean, linestyle="--", linewidth=1.5, label=f"baseline mean")
        # plt.axhline(baseline_median, linestyle=":", linewidth=1.8, label=f"baseline median")
        # plt.axhline(baseline_p, linestyle="-.", linewidth=1.5, label=f"baseline p{p}")

        
        plt.ylabel(f"Mahalanobis distance ({name})")
        plt.yscale("log")

        # Grid
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(rotation=45, ha="right")

        # # X ticks every 3 minutes
        # plt.xlabel("Synthetic Datetime (CET)")
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=3))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        # ---- custom excitation labels ----
        plt.xlabel("Excitation label")
        ax = plt.gca()

        dt_slots = np.unique(dt_sorted)  # unique synthetic time slots
        exc_unique = np.unique(exc_sorted)

        # case 1: only one excitation type
        if len(exc_unique) == 1:
            labels = [f"exc {int(exc_unique[0])}"] * len(dt_slots)
        # case 2: multiple excitations
        else:
            labels = []
            for t in dt_slots:
                # find samples belonging to this slot
                idx = (dt_sorted == t)
                # take the excitation label of that slot
                e = int(np.asarray(exc_sorted)[idx][0])
                labels.append(f"exc {e}")

        ax.set_xticks(dt_slots)
        ax.set_xticklabels(labels, rotation=45, ha="right")

        # Legend inside with 3 rows
        handles, labels = plt.gca().get_legend_handles_labels()
        n_items = len(labels)
        ncol = int(np.ceil(n_items / 6))
        plt.legend(handles, labels,
           ncol=ncol,
           markerscale=1.5,
           fontsize=9,
           loc='lower right')
        
        plt.tight_layout()
        plt.show()


    return mahal, baseline_p