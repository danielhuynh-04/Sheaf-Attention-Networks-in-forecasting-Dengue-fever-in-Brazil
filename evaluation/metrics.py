# evaluation/metrics.py
# ----------------------------------------------------------
# Core evaluation functions used by run_global_gat.py and
# tools/permutation_test.py for regression/classification metrics.
#
# Functions:
#   - evaluate_regression: MAE, RMSE, R2, SMAPE
#   - trimmed_r2: outlier-robust R2 (trim percentile tails)
#   - classification_metrics_from_regression: ROC-AUC, PR-AUC
#     from continuous regression predictions (threshold-based)
# ----------------------------------------------------------
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_regression(y_pred, y_true) -> dict:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    y_pred : array-like or torch.Tensor
        Predicted values (log-space or real-space).
    y_true : array-like or torch.Tensor
        Ground truth values.

    Returns
    -------
    dict with keys: MAE, RMSE, R2, SMAPE
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    denom = np.abs(y_true) + np.abs(y_pred)
    safe = denom > 0
    if safe.any():
        smape = float(np.mean(2.0 * np.abs(y_true[safe] - y_pred[safe]) / denom[safe]) * 100)
    else:
        smape = 0.0

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "SMAPE": smape}


def trimmed_r2(
    y_true,
    y_pred,
    trim: float = 0.01,
) -> float:
    """
    Outlier-robust R2 score by trimming extreme percentile tails.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    trim : float
        Fraction to trim from each tail (e.g. 0.01 = 1st–99th percentile).

    Returns
    -------
    float : trimmed R2 score
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_true) < 10:
        return float(r2_score(y_true, y_pred))

    lo = np.quantile(y_true, trim)
    hi = np.quantile(y_true, 1.0 - trim)
    mask = (y_true >= lo) & (y_true <= hi)

    if mask.sum() < 5:
        return float(r2_score(y_true, y_pred))

    return float(r2_score(y_true[mask], y_pred[mask]))


def classification_metrics_from_regression(
    y_true_real,
    y_pred_real,
    pos_threshold: float | None = None,
    q: float = 0.90,
) -> dict:
    """
    Derive binary classification metrics (ROC-AUC, PR-AUC) from
    continuous regression predictions by thresholding.

    An observation is classified as a positive (outbreak week) if
    y_true >= threshold, where threshold defaults to the q-th
    quantile of y_true (e.g. q=0.90 => 90th percentile).

    Parameters
    ----------
    y_true_real : array-like
        Ground truth in real-space (cases).
    y_pred_real : array-like
        Predictions in real-space (cases).
    pos_threshold : float or None
        Explicit threshold for positive class. If None, uses q-th
        quantile of y_true_real.
    q : float
        Quantile for automatic threshold derivation (default 0.90).

    Returns
    -------
    dict with keys: ROC_AUC, PR_AUC, pos_rate, threshold
    """
    y_true_real = np.asarray(y_true_real, dtype=np.float64).ravel()
    y_pred_real = np.asarray(y_pred_real, dtype=np.float64).ravel()

    if pos_threshold is None:
        pos_threshold = float(np.quantile(y_true_real, q))

    y_binary = (y_true_real >= pos_threshold).astype(int)
    pos_rate = float(y_binary.mean())

    # Edge case: all same class → AUC is undefined
    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return {
            "ROC_AUC": float("nan"),
            "PR_AUC": float("nan"),
            "pos_rate": pos_rate,
            "threshold": float(pos_threshold),
        }

    roc_auc = float(roc_auc_score(y_binary, y_pred_real))
    pr_auc = float(average_precision_score(y_binary, y_pred_real))

    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "pos_rate": pos_rate,
        "threshold": float(pos_threshold),
    }
