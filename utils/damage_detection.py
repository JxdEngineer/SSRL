
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def eval_damage_by_percentile(mahal, base_mask, dmg, percentile, healthy_label=1, name="mahal", print_metrics=True):
    """
    mahal: (N,) novelty score (e.g., Mahalanobis distances)
    base_mask: (N,) bool, True for baseline samples (used to compute threshold)
    dmg: (N,) damage labels, where healthy_label means undamaged
    percentile: e.g., 95
    healthy_label: label indicating undamaged (default 1)
    name: label for printing

    Returns: (cm, metrics_dict, thr)
    """
    mahal = np.asarray(mahal).reshape(-1)
    base_mask = np.asarray(base_mask).astype(bool).reshape(-1)
    dmg = np.asarray(dmg).reshape(-1)

    if not (mahal.shape[0] == base_mask.shape[0] == dmg.shape[0]):
        raise ValueError("mahal, base_mask, dmg must have the same length")

    # ground truth from damage labels: 0 = undamaged, 1 = damaged
    y_true = (dmg != healthy_label).astype(int)

    # threshold from baseline percentile
    if base_mask.sum() < 2:
        raise ValueError(f"[{name}] baseline has too few samples: {int(base_mask.sum())}")
    thr = float(np.percentile(mahal[base_mask], percentile))

    # predict damaged if mahal > thr
    y_pred = (mahal > thr).astype(int)

    # confusion matrix: rows=true, cols=pred, label order [undamaged(0), damaged(1)]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    

    # metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = None
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, mahal)

    if print_metrics:
        print(f"\n[{name}] threshold (baseline p{percentile}): {thr}")
        print("Confusion matrix (rows=true [undamaged, damaged], cols=pred [undamaged, damaged]):")
        print(cm)
        print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

        print("Metrics (positive class = damaged):")
        print(f"True negative rate          : {tn/(tn+fp):.4f}")
        print(f"True positive rate          : {tp/(tp+fn):.4f}")
        print(f"Accuracy          : {acc:.4f}")
        # print(f"Precision         : {prec:.4f}")
        # print(f"Recall            : {rec:.4f}")
        # print(f"F1                : {f1:.4f}")
        # if auc is not None:
        #     print(f"ROC-AUC ({name})     : {auc:.4f}")

    return cm, thr

def eval_damage_by_percentile_baseline_train(mahal, thr, dmg, percentile, healthy_label=1, name="mahal", print_metrics=True):
    """
    mahal: (N,) novelty score (e.g., Mahalanobis distances)
    thr: damage threshold, scalar
    percentile: percentile of the Mahalanobis distances of the baseline subset
    dmg: (N,) damage labels, where healthy_label means undamaged
    healthy_label: label indicating undamaged (default 1)
    name: label for printing

    Returns: (cm, metrics_dict, thr)
    """
    mahal = np.asarray(mahal).reshape(-1)
    dmg = np.asarray(dmg).reshape(-1)

    # ground truth from damage labels: 0 = undamaged, 1 = damaged
    y_true = (dmg != healthy_label).astype(int)

    # predict damaged if mahal > thr
    y_pred = (mahal > thr).astype(int)

    # confusion matrix: rows=true, cols=pred, label order [undamaged(0), damaged(1)]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    

    # metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = None
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, mahal)

    if print_metrics:
        print(f"\n[{name}] threshold (baseline p{percentile}): {thr}")
        print("Confusion matrix (rows=true [undamaged, damaged], cols=pred [undamaged, damaged]):")
        print(cm)
        print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

        print("Metrics (positive class = damaged):")
        print(f"True negative rate          : {tn/(tn+fp):.4f}")
        print(f"True positive rate          : {tp/(tp+fn):.4f}")
        print(f"Accuracy          : {acc:.4f}")
        print(f"Balanced Accuracy          : {bacc:.4f}")

    return cm