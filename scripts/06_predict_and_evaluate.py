from src.utils.io_utils import (
    load_yaml,
    read_jsonl,
    write_jsonl
)
from src.model.hmm_model import BetaHMM
from src.evaluation.metrics import (
    flatten_labels,
    compute_all_metrics,
    save_metrics_json,
    save_text,
    save_confusion_matrix_csv,
    plot_confusion_matrix,
    plot_metric_bars
)


def beta_segments(labels):
    segs = []
    start = None

    for i, lab in enumerate(labels):
        if lab == "B" and start is None:
            start = i
        elif lab != "B" and start is not None:
            segs.append((start, i - 1))
            start = None

    if start is not None:
        segs.append((start, len(labels) - 1))
    return segs


def main():
    cfg = load_yaml("configs/config.yaml")

    model = BetaHMM.load(cfg["model_json"])
    test_records = read_jsonl(cfg["test_jsonl"])

    predicted_records = []
    for rec in test_records:
        pred = model.predict(rec["sequence"])
        out = dict(rec)
        out["pred_labels"] = pred
        out["true_beta_segments"] = beta_segments(rec["labels"])
        out["pred_beta_segments"] = beta_segments(pred)
        predicted_records.append(out)

    write_jsonl(predicted_records, cfg["predictions_jsonl"])

    y_true, y_pred = flatten_labels(predicted_records)
    metrics, report_text, cm = compute_all_metrics(y_true, y_pred)

    save_metrics_json(metrics, cfg["metrics_json"])
    save_text(report_text, cfg["classification_report_txt"])
    save_confusion_matrix_csv(cm, cfg["confusion_matrix_csv"])
    plot_confusion_matrix(cm, cfg["confusion_matrix_png"])
    plot_metric_bars(metrics, cfg["metric_barplot_png"])

    summary = []
    summary.append("BETA SHEET HMM PROJECT SUMMARY")
    summary.append("=" * 40)
    summary.append(f"Test sequence count: {len(predicted_records)}")
    summary.append(f"Residue-level accuracy: {metrics['accuracy']:.4f}")
    summary.append(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    summary.append(f"MCC: {metrics['mcc']:.4f}")
    summary.append(f"Beta precision: {metrics['beta_precison']:.4f}")
    summary.append(f"Beta recall: {metrics['beta_recall']:.4f}")
    summary.append(f"Beta F1: {metrics['bata_f1']:.4f}")

    save_text("\n".join(summary), cfg["summary_text"])

    print("\n".join(summary))
    print("\nClassification report:\n")
    print(report_text)


if __name__ == "__main__":
    main()