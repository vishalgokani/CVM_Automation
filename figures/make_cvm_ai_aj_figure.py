"""Compare CSV columns AI and AJ and create CVM performance figures.

Column AI is expected to be ``predicted_CVM`` and column AJ is expected to be
``Ground_Truth`` in the measurement CSV.

Example:
    python figures/make_cvm_ai_aj_figure.py \
        --input "<MEASUREMENTS_CSV>" \
        --output-dir figures
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Pillow is required. Install it with: pip install pillow", file=sys.stderr)
    raise SystemExit(2)


VALID_GT_LABELS = ["CVM1", "CVM2", "CVM3", "CVM4-6"]
PREDICTED_LABELS = ["CVM1", "CVM2", "CVM3", "CVM4-6", "atypical"]
ORDERED_LABELS = ["CVM1", "CVM2", "CVM3", "CVM4-6"]

PANEL_SIZE = 175
PANEL_X = {"a": 28, "b": 227, "c": 426, "d": 625}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AI predicted_CVM vs AJ Ground_Truth and make figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input measurements CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where panel images, SVG, and summary files are saved.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        rows = list(reader)
        return reader.fieldnames, rows


def validate_columns(fieldnames: list[str]) -> tuple[str, str]:
    if len(fieldnames) < 36:
        raise ValueError(
            f"Expected at least 36 columns so AI/AJ exist; found {len(fieldnames)}."
        )

    ai_name = fieldnames[34]
    aj_name = fieldnames[35]
    if ai_name != "predicted_CVM" or aj_name != "Ground_Truth":
        raise ValueError(
            "Unexpected AI/AJ headers: "
            f"AI={ai_name!r}, AJ={aj_name!r}. "
            "Expected AI='predicted_CVM' and AJ='Ground_Truth'."
        )
    return ai_name, aj_name


def normalize(value: str | None) -> str:
    return (value or "").strip()


def confusion_matrix(
    rows: list[dict[str, str]], true_col: str, pred_col: str
) -> np.ndarray:
    matrix = np.zeros((len(VALID_GT_LABELS), len(PREDICTED_LABELS)), dtype=int)
    true_index = {label: idx for idx, label in enumerate(VALID_GT_LABELS)}
    pred_index = {label: idx for idx, label in enumerate(PREDICTED_LABELS)}

    for row in rows:
        true_label = normalize(row[true_col])
        pred_label = normalize(row[pred_col])
        if true_label in true_index and pred_label in pred_index:
            matrix[true_index[true_label], pred_index[pred_label]] += 1
    return matrix


def valid_ground_truth_rows(rows: list[dict[str, str]], true_col: str) -> list[dict[str, str]]:
    return [row for row in rows if normalize(row[true_col]) in VALID_GT_LABELS]


def paired_ordinal_rows(
    rows: list[dict[str, str]], true_col: str, pred_col: str
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if normalize(row[true_col]) in ORDERED_LABELS
        and normalize(row[pred_col]) in ORDERED_LABELS
    ]


def save_confusion_panel(matrix: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=220)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(PREDICTED_LABELS)), PREDICTED_LABELS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(VALID_GT_LABELS)), VALID_GT_LABELS)
    ax.set_xlabel("Predicted CVM")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_recall_panel(matrix: np.ndarray, output: Path) -> None:
    totals = matrix.sum(axis=1)
    correct = np.array([matrix[idx, idx] for idx in range(len(VALID_GT_LABELS))])
    recall = np.divide(correct, totals, out=np.zeros_like(correct, dtype=float), where=totals > 0)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=220)
    bars = ax.bar(VALID_GT_LABELS, recall, color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Per-stage recall")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    for bar, rate, total in zip(bars, recall, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 0.035, 0.98),
            f"{rate:.2f}\nn={total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_error_panel(rows: list[dict[str, str]], true_col: str, pred_col: str, output: Path) -> None:
    label_to_stage = {label: idx + 1 for idx, label in enumerate(ORDERED_LABELS)}
    errors: list[int] = []
    skipped = 0
    for row in rows:
        true_label = normalize(row[true_col])
        pred_label = normalize(row[pred_col])
        if true_label in label_to_stage and pred_label in label_to_stage:
            errors.append(label_to_stage[pred_label] - label_to_stage[true_label])
        elif true_label in label_to_stage:
            skipped += 1

    bins = np.arange(-3.5, 4.5, 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=220)
    ax.hist(errors, bins=bins, color="#F58518", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xticks(range(-3, 4))
    ax.set_xlabel("Predicted stage index - true stage index")
    ax.set_ylabel("Cases")
    ax.set_title("Signed ordinal error")
    if skipped:
        ax.text(
            0.02,
            0.96,
            f"{skipped} atypical/invalid predictions omitted",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
        )
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def ordinal_arrays(
    rows: list[dict[str, str]], true_col: str, pred_col: str
) -> tuple[np.ndarray, np.ndarray]:
    label_to_stage = {label: idx + 1 for idx, label in enumerate(ORDERED_LABELS)}
    true_values = [label_to_stage[normalize(row[true_col])] for row in rows]
    pred_values = [label_to_stage[normalize(row[pred_col])] for row in rows]
    return np.array(true_values, dtype=int), np.array(pred_values, dtype=int)


def quadratic_weighted_kappa(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    if true_values.size == 0:
        return float("nan")

    n_classes = len(ORDERED_LABELS)
    observed = np.zeros((n_classes, n_classes), dtype=float)
    for true_value, pred_value in zip(true_values, pred_values):
        observed[true_value - 1, pred_value - 1] += 1

    true_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / observed.sum()

    indices = np.arange(n_classes)
    weights = ((indices[:, None] - indices[None, :]) ** 2) / ((n_classes - 1) ** 2)
    observed_weighted = (weights * observed).sum()
    expected_weighted = (weights * expected).sum()
    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else float("nan")
    return 1.0 - observed_weighted / expected_weighted


def ordinal_metrics(rows: list[dict[str, str]], true_col: str, pred_col: str) -> dict[str, float | int]:
    true_values, pred_values = ordinal_arrays(rows, true_col, pred_col)
    if true_values.size == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "mean_absolute_ordinal_error": float("nan"),
            "median_absolute_ordinal_error": float("nan"),
            "within_one_stage_accuracy": float("nan"),
            "quadratic_weighted_kappa": float("nan"),
        }

    errors = pred_values - true_values
    abs_errors = np.abs(errors)
    recalls = []
    for stage in range(1, len(ORDERED_LABELS) + 1):
        mask = true_values == stage
        if mask.any():
            recalls.append(float(np.mean(pred_values[mask] == true_values[mask])))

    return {
        "n": int(true_values.size),
        "accuracy": float(np.mean(errors == 0)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "mean_absolute_ordinal_error": float(np.mean(abs_errors)),
        "median_absolute_ordinal_error": float(np.median(abs_errors)),
        "within_one_stage_accuracy": float(np.mean(abs_errors <= 1)),
        "quadratic_weighted_kappa": float(quadratic_weighted_kappa(true_values, pred_values)),
    }


def save_standalone_ordinal_panel(
    rows: list[dict[str, str]],
    true_col: str,
    pred_col: str,
    output: Path,
    show_metrics: bool = True,
    show_counts: bool = True,
) -> dict[str, float | int]:
    true_values, pred_values = ordinal_arrays(rows, true_col, pred_col)
    metrics = ordinal_metrics(rows, true_col, pred_col)
    errors = pred_values - true_values

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=450)
    bins = np.arange(-3.5, 4.5, 1)
    counts, _, patches = ax.hist(errors, bins=bins, color="#F58518", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xticks(range(-3, 4))
    ax.set_xlabel("Predicted stage index - true stage index")
    ax.set_ylabel("Paired ordered cases")
    ax.set_title("Signed ordinal error")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    if show_counts:
        for count, patch in zip(counts, patches):
            if count:
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    count + max(counts) * 0.025,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    if show_metrics:
        metric_text = (
            f"n = {metrics['n']}\n"
            f"QWK = {metrics['quadratic_weighted_kappa']:.3f}\n"
            f"MAOE = {metrics['mean_absolute_ordinal_error']:.2f}\n"
            f"Within 1 stage = {metrics['within_one_stage_accuracy']:.3f}\n"
            f"Exact accuracy = {metrics['accuracy']:.3f}\n"
            f"Balanced accuracy = {metrics['balanced_accuracy']:.3f}"
        )
        ax.text(
            0.98,
            0.96,
            metric_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#777777"},
        )

    fig.tight_layout()
    save_kwargs = {"bbox_inches": "tight"}
    if output.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update({"pil_kwargs": {"quality": 95, "subsampling": 0}})
    fig.savefig(output, **save_kwargs)
    plt.close(fig)
    return metrics


def save_true_pred_panel(
    rows: list[dict[str, str]], true_col: str, pred_col: str, output: Path
) -> None:
    y_labels = ["atypical", *ORDERED_LABELS]
    true_to_x = {label: idx + 1 for idx, label in enumerate(ORDERED_LABELS)}
    pred_to_y = {label: idx for idx, label in enumerate(y_labels)}

    xs: list[float] = []
    ys: list[float] = []
    colors: list[str] = []
    valid_rows = [row for row in rows if normalize(row[true_col]) in true_to_x]

    for idx, row in enumerate(valid_rows):
        true_label = normalize(row[true_col])
        pred_label = normalize(row[pred_col])
        if pred_label not in pred_to_y:
            continue
        jitter = ((idx % 9) - 4) * 0.018
        xs.append(true_to_x[true_label] + jitter)
        ys.append(pred_to_y[pred_label] + jitter)
        colors.append("#54A24B" if pred_label == true_label else "#E45756")

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=220)
    ax.scatter(xs, ys, s=18, c=colors, alpha=0.72, linewidths=0)
    ax.plot([1, 4], [1, 4], color="#555555", linewidth=1, linestyle="--")
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks(range(1, 5), ORDERED_LABELS, rotation=25, ha="right")
    ax.set_yticks(range(0, 5), y_labels)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted CVM")
    ax.set_title("True vs predicted")
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def image_to_png_data_uri(path: Path) -> str:
    with Image.open(path) as source:
        source.load()
        if source.mode not in ("RGB", "RGBA"):
            source = source.convert("RGBA" if "transparency" in source.info else "RGB")
        buffer = io.BytesIO()
        source.save(buffer, format="PNG", optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_svg(image_paths: dict[str, Path]) -> str:
    images = {letter: image_to_png_data_uri(path) for letter, path in image_paths.items()}
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="8.0in" height="2.05in" viewBox="0 0 800 205">',
        "  <title>CVM AI versus AJ comparison figure</title>",
        '  <rect x="0" y="0" width="800" height="205" fill="white"/>',
        "  <style>",
        "    .panel-label {",
        '      font-family: "Times New Roman", Times, serif;',
        "      font-size: 22px; font-weight: bold; fill: black;",
        "    }",
        "    .panel-border { fill: none; stroke: #777777; stroke-width: 1.2; }",
        "  </style>",
    ]

    for index, letter in enumerate("abcd"):
        panel_x = PANEL_X[letter]
        label_x = 8 + index * 199
        elements.extend(
            [
                f"  <!-- Panel {letter.upper()}: {image_paths[letter].name} -->",
                f'  <text class="panel-label" x="{label_x}" y="22">{letter.upper()}</text>',
                f'  <image x="{panel_x}" y="12" width="{PANEL_SIZE}" '
                f'height="{PANEL_SIZE}" preserveAspectRatio="xMidYMid meet" '
                f'href="{images[letter]}"/>',
                f'  <rect class="panel-border" x="{panel_x}" y="12" '
                f'width="{PANEL_SIZE}" height="{PANEL_SIZE}"/>',
            ]
        )

    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def summarize(
    rows: list[dict[str, str]], true_col: str, pred_col: str, matrix: np.ndarray
) -> dict[str, object]:
    valid_rows = valid_ground_truth_rows(rows, true_col)
    exact_matches = sum(
        1 for row in valid_rows if normalize(row[pred_col]) == normalize(row[true_col])
    )
    invalid_ground_truth = Counter(
        normalize(row[true_col])
        for row in rows
        if normalize(row[true_col]) not in VALID_GT_LABELS
    )
    prediction_counts = Counter(normalize(row[pred_col]) for row in rows)
    ground_truth_counts = Counter(normalize(row[true_col]) for row in rows)

    totals = matrix.sum(axis=1)
    correct = np.array([matrix[idx, idx] for idx in range(len(VALID_GT_LABELS))])
    recall = np.divide(correct, totals, out=np.zeros_like(correct, dtype=float), where=totals > 0)

    return {
        "total_rows": len(rows),
        "valid_ground_truth_rows": len(valid_rows),
        "exact_matches_on_valid_ground_truth": exact_matches,
        "accuracy_on_valid_ground_truth": exact_matches / len(valid_rows) if valid_rows else None,
        "ground_truth_counts": dict(sorted(ground_truth_counts.items())),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "invalid_or_blank_ground_truth_counts": dict(sorted(invalid_ground_truth.items())),
        "per_stage_recall": {
            label: {
                "correct": int(correct[idx]),
                "total": int(totals[idx]),
                "recall": float(recall[idx]),
            }
            for idx, label in enumerate(VALID_GT_LABELS)
        },
        "confusion_matrix": {
            "rows": VALID_GT_LABELS,
            "columns": PREDICTED_LABELS,
            "counts": matrix.tolist(),
        },
    }


def write_summary(summary: dict[str, object], output: Path) -> None:
    accuracy = summary["accuracy_on_valid_ground_truth"]
    lines = [
        "AI vs AJ comparison",
        "===================",
        "",
        "AI column: predicted_CVM",
        "AJ column: Ground_Truth",
        f"Total rows: {summary['total_rows']}",
        f"Rows with valid CVM ground truth: {summary['valid_ground_truth_rows']}",
        (
            "Exact matches on valid ground truth: "
            f"{summary['exact_matches_on_valid_ground_truth']}"
        ),
        (
            "Accuracy on valid ground truth: "
            f"{accuracy:.3f}" if isinstance(accuracy, float) else "Accuracy on valid ground truth: n/a"
        ),
        "",
        "Ground-truth counts:",
    ]
    for label, count in summary["ground_truth_counts"].items():
        printable = "<blank>" if label == "" else label
        lines.append(f"  {printable}: {count}")

    lines.extend(["", "Prediction counts:"])
    for label, count in summary["prediction_counts"].items():
        printable = "<blank>" if label == "" else label
        lines.append(f"  {printable}: {count}")

    lines.extend(["", "Invalid or blank AJ values excluded from performance panels:"])
    for label, count in summary["invalid_or_blank_ground_truth_counts"].items():
        printable = "<blank>" if label == "" else label
        lines.append(f"  {printable}: {count}")

    lines.extend(["", "Per-stage recall:"])
    for label, values in summary["per_stage_recall"].items():
        lines.append(
            f"  {label}: {values['correct']}/{values['total']} = {values['recall']:.3f}"
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_row_comparison(
    rows: list[dict[str, str]], true_col: str, pred_col: str, output: Path
) -> None:
    fieldnames = [
        "filename",
        "predicted_CVM",
        "Ground_Truth",
        "ground_truth_status",
        "exact_match",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            true_label = normalize(row[true_col])
            pred_label = normalize(row[pred_col])
            gt_status = "valid_cvm" if true_label in VALID_GT_LABELS else "excluded"
            writer.writerow(
                {
                    "filename": normalize(row.get("filename")),
                    "predicted_CVM": pred_label,
                    "Ground_Truth": true_label,
                    "ground_truth_status": gt_status,
                    "exact_match": (
                        "yes" if gt_status == "valid_cvm" and pred_label == true_label else "no"
                    ),
                }
            )


def write_ordinal_metrics(metrics: dict[str, float | int], output: Path) -> None:
    lines = [
        "Paired ordinal AI vs AJ metrics",
        "===============================",
        "",
        "Included rows: AJ in CVM1/CVM2/CVM3/CVM4-6 and AI in CVM1/CVM2/CVM3/CVM4-6.",
        "Excluded rows: blank AJ, non-ordered AJ values, and AI atypical predictions.",
        "Ordinal order: CVM1, CVM2, CVM3, CVM4-6.",
        "",
        f"n: {metrics['n']}",
        f"Quadratic weighted kappa: {metrics['quadratic_weighted_kappa']:.3f}",
        f"Mean absolute ordinal error: {metrics['mean_absolute_ordinal_error']:.3f}",
        f"Median absolute ordinal error: {metrics['median_absolute_ordinal_error']:.3f}",
        f"Within-one-stage accuracy: {metrics['within_one_stage_accuracy']:.3f}",
        f"Exact accuracy: {metrics['accuracy']:.3f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        fieldnames, rows = load_rows(args.input.expanduser())
        pred_col, true_col = validate_columns(fieldnames)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_rows = valid_ground_truth_rows(rows, true_col)
        ordinal_rows = paired_ordinal_rows(rows, true_col, pred_col)
        matrix = confusion_matrix(rows, true_col, pred_col)

        panel_paths = {
            "a": output_dir / "cvm_ai_aj_confusion_matrix.png",
            "b": output_dir / "cvm_ai_aj_per_stage_recall.png",
            "c": output_dir / "cvm_ai_aj_signed_ordinal_error.png",
            "d": output_dir / "cvm_ai_aj_true_vs_predicted.png",
        }
        save_confusion_panel(matrix, panel_paths["a"])
        save_recall_panel(matrix, panel_paths["b"])
        save_error_panel(valid_rows, true_col, pred_col, panel_paths["c"])
        save_true_pred_panel(valid_rows, true_col, pred_col, panel_paths["d"])
        ordinal_metrics_summary = save_standalone_ordinal_panel(
            ordinal_rows,
            true_col,
            pred_col,
            output_dir / "cvm_ai_aj_signed_ordinal_metrics.png",
        )
        save_standalone_ordinal_panel(
            ordinal_rows,
            true_col,
            pred_col,
            output_dir / "cvm_ai_aj_signed_ordinal_metrics.svg",
        )
        for suffix in ("png", "svg", "jpg"):
            save_standalone_ordinal_panel(
                ordinal_rows,
                true_col,
                pred_col,
                output_dir / f"cvm_ai_aj_signed_ordinal_clean.{suffix}",
                show_metrics=False,
                show_counts=False,
            )

        svg_path = output_dir / "cvm_ai_aj_four_panel_figure.svg"
        svg_path.write_text(build_svg(panel_paths), encoding="utf-8")

        summary = summarize(rows, true_col, pred_col, matrix)
        write_summary(summary, output_dir / "cvm_ai_aj_summary.txt")
        write_ordinal_metrics(
            ordinal_metrics_summary,
            output_dir / "cvm_ai_aj_signed_ordinal_metrics.txt",
        )
        write_row_comparison(rows, true_col, pred_col, output_dir / "cvm_ai_aj_row_comparison.csv")
        (output_dir / "cvm_ai_aj_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved outputs in: {output_dir}")
    print(f"Saved four-panel SVG: {svg_path}")
    print(f"Rows with valid CVM ground truth: {len(valid_rows)} / {len(rows)}")
    print(f"Rows with paired ordered AI/AJ labels: {len(ordinal_rows)} / {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
