
import torch

def compute_pairwise_agreement_metrics(true_labels, pred_labels):
    """
    Compute pairwise precision and recall for cluster agreement.

    Recall: Of all pairs that should be from the same source image,
            what fraction were correctly predicted to be together?

    Precision: Of all pairs predicted to be in the same cluster,
               what fraction are actually from the same source image?

    Args:
        true_labels: Ground truth image source labels (n,)
        pred_labels: Predicted cluster labels (n,)

    Returns:
        Dictionary with precision, recall, f1_score, and pair counts
    """
    n_points = len(true_labels)

    # Convert to numpy for easier processing
    true_labels = true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else true_labels
    pred_labels = pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else pred_labels

    # Count different types of pairs
    true_positive = 0  # Same cluster in both true and predicted
    false_positive = 0  # Same cluster in predicted but different in true
    false_negative = 0  # Different cluster in predicted but same in true
    true_negative = 0  # Different cluster in both true and predicted

    # Check all pairs (i, j) where i < j
    for i in range(n_points):
        for j in range(i + 1, n_points):
            same_true = (true_labels[i] == true_labels[j])
            same_pred = (pred_labels[i] == pred_labels[j])

            if same_true and same_pred:
                true_positive += 1
            elif same_pred and not same_true:
                false_positive += 1
            elif same_true and not same_pred:
                false_negative += 1
            else:  # not same_true and not same_pred
                true_negative += 1

    # Compute metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_negative': true_negative,
        'total_pairs': n_points * (n_points - 1) // 2
    }