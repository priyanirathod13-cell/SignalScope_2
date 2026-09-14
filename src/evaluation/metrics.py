import numpy as np

def compute_roc_auc(y_true, y_scores):
    """
    Computes exact Area Under the ROC Curve (ROC-AUC)
    using the Wilcoxon-Mann-Whitney rank-sum statistic.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
        
    order = np.argsort(y_scores)
    ranks = np.empty(len(y_scores), dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1)
    
    pos_ranks = ranks[y_true == 1]
    n_pos = len(pos)
    n_neg = len(neg)
    
    u = np.sum(pos_ranks) - (n_pos * (n_pos + 1)) / 2
    return float(u / (n_pos * n_neg))

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Computes complete classification metrics:
    Accuracy, Precision, Recall, F1 (binary & macro), ROC-AUC, and Confusion Matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = float(np.mean(y_true == y_pred))
    
    # Class 1 (Synthetic) metrics
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0

    # Class 0 (Real) metrics
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0

    macro_f1 = (f1_0 + f1_1) / 2
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2

    # ROC-AUC (using probability of class 1)
    auc = compute_roc_auc(y_true, y_prob)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(auc, 4),
        "f1": round(macro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "synthetic_f1": round(f1_1, 4),
        "real_f1": round(f1_0, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    }

def format_confusion_matrix(cm_dict):
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]
    
    lines = [
        "               Predicted Real (0)   Predicted Synthetic (1)",
        f"Actual Real (0)        {tn:<10d}           {fp:<10d}",
        f"Actual Synth (1)       {fn:<10d}           {tp:<10d}"
    ]
    return "\n".join(lines)
