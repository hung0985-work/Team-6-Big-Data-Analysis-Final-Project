
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    brier_score_loss
)


# ------------------------------------------------------------
# Function 1. Evaluate probability-based predictions
# ------------------------------------------------------------

def evaluate_predictions(y_true, y_prob, threshold=0.5):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),

        "average_precision": average_precision_score(
            y_true, y_prob
        ),

        "roc_auc": roc_auc_score(
            y_true, y_prob
        ),

        "accuracy": accuracy_score(
            y_true, y_pred
        ),

        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),

        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),

        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "specificity": specificity,

        "brier_score": brier_score_loss(
            y_true, y_prob
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }


# ------------------------------------------------------------
# Function 2. Optimize classification threshold
# ------------------------------------------------------------

def optimize_f1_threshold(y_true, y_prob):

    precision, recall, thresholds = precision_recall_curve(
        y_true, y_prob
    )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0
    )

    best_idx = np.argmax(f1_values)

    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "optimal_f1": float(f1_values[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx])
    }


# ------------------------------------------------------------
# Function 3. Summarize cross-validation performance
# ------------------------------------------------------------

def summarize_cv_results(cv_results):

    results_df = pd.DataFrame(cv_results)

    metric_columns = [
        "average_precision",
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "brier_score",
        "training_time_sec",
        "inference_time_sec"
    ]

    available_metrics = [
        col for col in metric_columns
        if col in results_df.columns
    ]

    summary = (
        results_df[available_metrics]
        .agg(["mean", "std"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    return summary


# ------------------------------------------------------------
# Function 4. Calculate balanced class weights
# ------------------------------------------------------------

def calculate_class_weights(y_train):

    y_train = np.asarray(y_train)

    classes, counts = np.unique(
        y_train,
        return_counts=True
    )

    n_samples = len(y_train)
    n_classes = len(classes)

    return {
        int(cls): float(
            n_samples / (n_classes * count)
        )
        for cls, count in zip(classes, counts)
    }


# ------------------------------------------------------------
# Function 5. Generate balanced sample weights
# ------------------------------------------------------------

def calculate_sample_weights(y_train):

    y_train = np.asarray(y_train)

    class_weights = calculate_class_weights(y_train)

    return np.asarray(
        [class_weights[int(label)] for label in y_train],
        dtype=np.float32
    )
