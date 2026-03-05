# utils/plot_latent.py

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go


ArrayLike = Union[np.ndarray, "object"]  # torch.Tensor also works via np.asarray


def _to_numpy(x: ArrayLike) -> np.ndarray:
    # supports numpy arrays, torch tensors, and lists
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


def _plot_3d_plotly(X3: np.ndarray, labels: np.ndarray, title: str, prefix: str) -> None:
    """
    X3: (N, 3)
    labels: (N,)
    """
    fig = go.Figure()
    labs = np.unique(labels)
    for lab in labs:
        idx = labels == lab
        fig.add_trace(
            go.Scatter3d(
                x=X3[idx, 0],
                y=X3[idx, 1],
                z=X3[idx, 2],
                mode="markers",
                name=f"{prefix} {int(lab)}",
                marker=dict(size=4),
            )
        )
    fig.update_layout(
        title=title,
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis_title=f"{title.split()[0]} 1",
            yaxis_title=f"{title.split()[0]} 2",
            zaxis_title=f"{title.split()[0]} 3",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


def plot_latent_2d(
    Z: ArrayLike,
    name: str,
    dmg: ArrayLike,
    exc: ArrayLike,
    labels_dt: Optional[ArrayLike] = None,
    day_order: Sequence[str] = ("5 May", "6 May", "7 May"),
    tsne_perplexity_cap: int = 30,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.05,
    random_state: int = 0,
) -> None:
    """
    Make 2D t SNE and 2D UMAP plots for a latent Z, colored by damage, excitation, and optionally day.

    Z: (N, C, Lz) or (N, D)
    dmg: (N,)
    exc: (N,)
    labels_dt: (N,) strings like "5 May", "6 May", "7 May", optional
    """
    H = _reduce_latent(Z)
    dmg = _to_numpy(dmg).reshape(-1)
    exc = _to_numpy(exc).reshape(-1)
    if labels_dt is not None:
        labels_dt = _to_numpy(labels_dt).reshape(-1)

    # # 2D t SNE
    # perplexity = min(tsne_perplexity_cap, max(5, (H.shape[0] - 1) // 3))
    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=perplexity,
    #     learning_rate="auto",
    #     init="pca",
    #     random_state=random_state,
    # )
    # H_2d = tsne.fit_transform(H)

    # # by damage
    # plt.figure()
    # for d in np.unique(dmg):
    #     idx = dmg == d
    #     plt.scatter(H_2d[idx, 0], H_2d[idx, 1], s=15, label=f"damage {int(d)}")
    # plt.xlabel("t SNE 1")
    # plt.ylabel("t SNE 2")
    # plt.title(f"t SNE of latent {name} (colored by damage)")
    # plt.legend(markerscale=1.5, fontsize=9)
    # plt.tight_layout()
    # plt.show()

    # # by excitation
    # plt.figure()
    # for e in np.unique(exc):
    #     idx = exc == e
    #     plt.scatter(H_2d[idx, 0], H_2d[idx, 1], s=15, label=f"excitation {int(e)}")
    # plt.xlabel("t SNE 1")
    # plt.ylabel("t SNE 2")
    # plt.title(f"t SNE of latent {name} (colored by excitation)")
    # plt.legend(markerscale=1.5, fontsize=9)
    # plt.tight_layout()
    # plt.show()

    # # by day
    # if labels_dt is not None:
    #     plt.figure()
    #     for lab in day_order:
    #         idx = labels_dt == lab
    #         if idx.sum() == 0:
    #             continue
    #         plt.scatter(H_2d[idx, 0], H_2d[idx, 1], s=15, label=str(lab))
    #     plt.xlabel("t SNE 1")
    #     plt.ylabel("t SNE 2")
    #     plt.title(f"t SNE of latent {name} (by day)")
    #     plt.legend(markerscale=1.5, fontsize=9)
    #     plt.tight_layout()
    #     plt.show()

    # 2D UMAP
    um = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    U2 = um.fit_transform(H)

    mpl.rcParams.update({"font.size": 10.5})

    # by damage
    plt.figure(figsize=(5, 5))
    for d in np.unique(dmg):
        idx = dmg == d
        plt.scatter(U2[idx, 0], U2[idx, 1], s=10, label=f"damage {int(d)}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"UMAP of latent {name} (by damage)")
    plt.legend(fontsize=9, markerscale=1.0)
    plt.tight_layout()
    plt.show()

    # by excitation
    plt.figure(figsize=(5, 5))
    for e in np.unique(exc):
        idx = exc == e
        plt.scatter(U2[idx, 0], U2[idx, 1], s=10, label=f"excitation {int(e)}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"UMAP of latent {name} (by excitation)")
    plt.legend(fontsize=9, markerscale=1.0)
    plt.tight_layout()
    plt.show()

    # by day
    if labels_dt is not None:
        plt.figure(figsize=(8, 6))
        for lab in day_order:
            idx = labels_dt == lab
            if idx.sum() == 0:
                continue
            plt.scatter(U2[idx, 0], U2[idx, 1], s=10, label=str(lab))
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"UMAP of latent {name} (by day)")
        plt.legend(markerscale=1.0, fontsize=9)
        plt.tight_layout()
        plt.show()


def plot_latent_3d(
    Z: ArrayLike,
    name: str,
    dmg: ArrayLike,
    exc: ArrayLike,
    labels_dt: Optional[ArrayLike] = None,
    day_order: Sequence[str] = ("5 May", "6 May", "7 May"),
    tsne_perplexity_cap: int = 30,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.4,
    random_state: int = 0,
) -> None:
    """
    Make 3D t SNE and 3D UMAP plots for a latent Z, colored by damage, excitation, and optionally day.
    Uses Plotly for interactive 3D plots.

    Z: (N, C, Lz) or (N, D)
    dmg: (N,)
    exc: (N,)
    labels_dt: (N,) strings like "5 May", "6 May", "7 May", optional
    """
    H = _reduce_latent(Z)
    dmg = _to_numpy(dmg).reshape(-1)
    exc = _to_numpy(exc).reshape(-1)
    if labels_dt is not None:
        labels_dt = _to_numpy(labels_dt).reshape(-1)

    # 3D t SNE
    # perplexity = min(tsne_perplexity_cap, max(5, (H.shape[0] - 1) // 3))
    # tsne = TSNE(
    #     n_components=3,
    #     perplexity=perplexity,
    #     learning_rate="auto",
    #     init="pca",
    #     random_state=random_state,
    # )
    # H_3d_tsne = tsne.fit_transform(H)

    # _plot_3d_plotly(H_3d_tsne, dmg, f"t SNE 3D of latent {name} (by damage)", "damage")
    # _plot_3d_plotly(H_3d_tsne, exc, f"t SNE 3D of latent {name} (by excitation)", "excitation")

    # if labels_dt is not None:
    #     day_to_int = {lab: i for i, lab in enumerate(day_order)}
    #     labels_dt_int = np.array([day_to_int.get(str(lab), -1) for lab in labels_dt], dtype=int)
    #     mask_day = labels_dt_int >= 0
    #     _plot_3d_plotly(
    #         H_3d_tsne[mask_day],
    #         labels_dt_int[mask_day],
    #         f"t SNE 3D of latent {name} (by day)",
    #         "day",
    #     )

    # 3D UMAP
    um = umap.UMAP(
        n_components=3,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="mahalanobis",
        random_state=random_state,
    )
    H_3d_umap = um.fit_transform(H)

    _plot_3d_plotly(H_3d_umap, dmg, f"UMAP 3D of latent {name} (by damage)", "damage")
    _plot_3d_plotly(H_3d_umap, exc, f"UMAP 3D of latent {name} (by excitation)", "excitation")

    if labels_dt is not None:
        day_to_int = {lab: i for i, lab in enumerate(day_order)}
        labels_dt_int = np.array([day_to_int.get(str(lab), -1) for lab in labels_dt], dtype=int)
        mask_day = labels_dt_int >= 0
        _plot_3d_plotly(
            H_3d_umap[mask_day],
            labels_dt_int[mask_day],
            f"UMAP 3D of latent {name} (by day)",
            "day",
        )