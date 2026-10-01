from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


def evaluate_cancellation_model(y_true: np.ndarray, probabilities: np.ndarray, baseline_rate: float) -> dict:
    """Evaluate an imbalanced cancellation model without an arbitrary 0.5 threshold."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(y_true)
    k = max(1, int(np.ceil(n * 0.10)))
    order = np.argsort(-probabilities)
    top_idx = order[:k]
    top_pred = np.zeros(n, dtype=int)
    top_pred[top_idx] = 1
    top_precision = precision_score(y_true, top_pred, zero_division=0)
    top_recall = recall_score(y_true, top_pred, zero_division=0)
    ap = average_precision_score(y_true, probabilities)
    roc = roc_auc_score(y_true, probabilities) if np.unique(y_true).size == 2 else float("nan")
    lift = float(top_precision / baseline_rate) if baseline_rate > 0 else float("nan")
    return {
        "top10_precision": float(top_precision),
        "top10_recall": float(top_recall),
        "average_precision": float(ap),
        "roc_auc": float(roc),
        "baseline_rate": float(baseline_rate),
        "top10_lift": lift,
    }


def pilot_impact_table(eval_df):
    """Return the operational review trade-off at 5/10/20/30% review shares."""
    ranked = eval_df.sort_values("CancellationProbability", ascending=False).reset_index(drop=True)
    total = ranked["CancelledOutcome"].sum()
    rows = []
    for share in [0.05, 0.10, 0.20, 0.30]:
        k = max(1, int(len(ranked) * share))
        caught = int(ranked.head(k)["CancelledOutcome"].sum())
        rows.append({
            "Review share": f"{share:.0%}",
            "Orders reviewed": k,
            "Cancellations caught": caught,
            "% of all cancellations caught": round(100 * caught / max(total, 1), 1),
            "Reviews per cancellation caught": round(k / max(caught, 1), 1),
        })
    return __import__("pandas").DataFrame(rows)
