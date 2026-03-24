import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    balanced_accuracy_score
)


def flatten_labels(records):
    y_true = []
    y_pred = []
    for rec in records:
        y_true.extend(rec["labels"])
        y_pred.extend(rec["pred_labels"])
    return y_true, y_pred


def compute_all_metrics(y_true, y_pred):
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=["B", "N"],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=["B", "N"])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "beta_precision": report_dict["B"]["precision"],
        "beta_recall": report_dict["B"]["recall"],
        "beta_f1": report_dict["B"]["f1-score"],
        "nonbeta_precision": report_dict["N"]["precision"],
        "nonbeta_recall": report_dict["N"]["recall"],
        "nonbeta_f1": report_dict["N"]["f1-score"],
        "support_beta": report_dict["B"]["support"],
        "support_nonbeta": report_dict["N"]["support"]
    }

    report_text = classification_report(
        y_true,
        y_pred,
        labels=["B", "N"],
        zero_division=0
    )

    return metrics, report_text, cm


def save_confusion_matrix_csv(cm, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", "B", "N"])
        writer.writerow(["B", cm[0, 0], cm[0, 1]])
        writer.writerow(["N", cm[1, 0], cm[1, 1]])


def save_metrics_json(metrics, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_text(text, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_confusion_matrix(cm, out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred B", "Pred N"])
    ax.set_yticklabels(["True B", "True N"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_metric_bars(metrics, out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    names = ["accuracy", "balanced_accuracy", "mcc", "beta_precision", "beta_recall", "beta_f1"]
    values = [metrics[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, values)
    ax.set_ylim(0, 1)
    ax.set_title("Model Metrics")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()