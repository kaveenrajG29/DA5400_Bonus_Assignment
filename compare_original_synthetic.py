#!/usr/bin/env python3
"""Compare test accuracy when training on real data vs condensed synthetic data."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from model.dcddm_models import Dualmodel
from module.config import config_to_args, parse_args, set_seed
from module.data import load_data, make_loaders
from module.evaluation import epoch, logits_from_output, metric_dict_from_arrays


DEFAULT_MODELS = ["MLP", "CNNBN", "CNNIN", "TCN"]


def load_synthetic_dataset(path: Path):
    archive = np.load(path, allow_pickle=False)
    if "x" not in archive.files or "y" not in archive.files:
        raise ValueError(f"{path} must contain arrays named 'x' and 'y'.")
    x = torch.tensor(archive["x"], dtype=torch.float32)
    y = torch.tensor(archive["y"], dtype=torch.long)
    if x.ndim != 3:
        raise ValueError(f"Expected synthetic x to have shape [N, C, T], got {tuple(x.shape)}.")
    return x, y


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def evaluate_with_predictions(net, test_loader, args):
    net.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x_true, y_true in test_loader:
            x = x_true.float().to(args.device)
            y = y_true.long().to(args.device)
            logits = logits_from_output(net(x))
            probs.append(F.softmax(logits, dim=1).cpu())
            labels.append(y.cpu())
    prob_np = torch.cat(probs).numpy()
    y_np = torch.cat(labels).numpy()
    metrics = metric_dict_from_arrays(y_np, prob_np, args.num_classes)
    return metrics, y_np, prob_np.argmax(axis=1)


def save_confusion_matrix(path: Path, y_true, y_pred, class_names, title: str):
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def sample_amplitude(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 2 and arr.shape[0] >= 2:
        return np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    if arr.ndim == 2:
        return np.abs(arr[0])
    return np.abs(arr)


def save_amplitude_plots(train, synth_x, synth_y, class_names, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, class_name in enumerate(class_names):
        real_indices = torch.nonzero(train["labels"] == label, as_tuple=False).flatten()
        synthetic_indices = torch.nonzero(synth_y == label, as_tuple=False).flatten()
        if len(real_indices) == 0 or len(synthetic_indices) == 0:
            continue
        real_amp = sample_amplitude(train["samples"][real_indices[0]])
        synthetic_amp = sample_amplitude(synth_x[synthetic_indices[0]])
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(real_amp, linewidth=1.0, color="#2563eb")
        axes[0].set_title(f"Original amplitude - {class_name}")
        axes[0].set_ylabel("Amplitude")
        axes[1].plot(synthetic_amp, linewidth=1.0, color="#dc2626")
        axes[1].set_title(f"Synthetic amplitude - {class_name}")
        axes[1].set_xlabel("Time index")
        axes[1].set_ylabel("Amplitude")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_name(class_name)}_amplitude_comparison.png", dpi=220)
        plt.close(fig)


def train_and_test(train_loader, test_loader, config, run_id: int, label: str, class_names, confusion_dir: Path):
    set_seed(config.seed + run_id)
    args = config_to_args(config)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    net = Dualmodel(args).to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=float(config.lr_teacher), momentum=config.mom, weight_decay=config.l2)
    time0 = time.time()
    train_loss = 0.0
    progress = tqdm(
        range(config.epoch_eval_train + 1),
        desc=f"{label} run {run_id}",
        leave=False,
        dynamic_ncols=True,
    )
    for ep in progress:
        train_loss, train_metrics = epoch(
            f"train_{label}",
            {"data_loader": train_loader, "model": config.model},
            net,
            optimizer,
            criterion,
            args,
            aug=True,
        )
        progress.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_metrics['Accuracy']:.4f}")
        if ep == 0 or ep == config.epoch_eval_train:
            print(
                f"{label} run {run_id}, epoch {ep}/{config.epoch_eval_train}, "
                f"train loss {train_loss:.6f}, accuracy {train_metrics['Accuracy']:.4f}"
            )
    test_metrics, y_true, y_pred = evaluate_with_predictions(net, test_loader, args)
    print(
        f"{label} run {run_id} finished in {time.time() - time0:.1f}s, "
        f"test accuracy {test_metrics['Accuracy']:.4f}"
    )
    save_confusion_matrix(
        confusion_dir / f"{safe_name(label)}_run{run_id}.png",
        y_true,
        y_pred,
        class_names,
        f"{label} run {run_id}",
    )
    return test_metrics


def summarize(runs):
    keys = runs[0].keys()
    return {
        key: {
            "mean": float(np.nanmean([run[key] for run in runs])),
            "std": float(np.nanstd([run[key] for run in runs])),
        }
        for key in keys
    }


def parse_model_list(value: str):
    if value.lower() == "paper":
        return DEFAULT_MODELS
    return [model.strip() for model in value.split(",") if model.strip()]


def format_stat(stats, key):
    mean = stats[key]["mean"]
    std = stats[key]["std"]
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def table_rows(results):
    rows = [
        [
            "Model",
            "Original Accuracy (%)",
            "Synthetic Accuracy (%)",
            "Accuracy Gap (%)",
            "Original F1 (%)",
            "Synthetic F1 (%)",
            "Original AUROC (%)",
            "Synthetic AUROC (%)",
            "Original AUPRC (%)",
            "Synthetic AUPRC (%)",
        ]
    ]
    for item in results:
        original = item["original"]
        synthetic = item["synthetic"]
        gap = (original["Accuracy"]["mean"] - synthetic["Accuracy"]["mean"]) * 100
        rows.append(
            [
                item["model"],
                format_stat(original, "Accuracy"),
                format_stat(synthetic, "Accuracy"),
                f"{gap:.2f}",
                format_stat(original, "F1"),
                format_stat(synthetic, "F1"),
                format_stat(original, "AUROC"),
                format_stat(synthetic, "AUROC"),
                format_stat(original, "AUPRC"),
                format_stat(synthetic, "AUPRC"),
            ]
        )
    return rows


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            escaped = []
            for cell in row:
                text = str(cell).replace('"', '""')
                escaped.append(f'"{text}"')
            f.write(",".join(escaped) + "\n")


def col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def save_xlsx(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_data = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, cell in enumerate(row, start=1):
            ref = f"{col_name(c_idx)}{r_idx}"
            text = escape(str(cell))
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_data.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_data)}</sheetData>'
        "</worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Accuracy Table" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", rels)
        xlsx.writestr("xl/workbook.xml", workbook)
        xlsx.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        xlsx.writestr("xl/worksheets/sheet1.xml", worksheet)


def main() -> None:
    config = parse_args()
    synthetic_path = config.synthetic_path or (config.output_dir / "condensed_dataset.npz")
    output_path = config.comparison_output or (config.output_dir / "accuracy_comparison.json")
    table_base = config.comparison_table or (config.output_dir / "accuracy_table.xlsx")

    train, val, test, class_names = load_data(config)
    loaders = make_loaders(train, val, test, config)

    synth_x, synth_y = load_synthetic_dataset(synthetic_path)
    if synth_x.shape[1:] != train["samples"].shape[1:]:
        raise ValueError(
            "Synthetic data shape does not match the loaded real data. "
            f"synthetic={tuple(synth_x.shape)}, real_sample={tuple(train['samples'].shape)}"
        )
    synthetic_loader = DataLoader(
        TensorDataset(synth_x.to(config.device), synth_y.to(config.device)),
        batch_size=config.batch_train,
        shuffle=True,
    )
    confusion_dir = config.output_dir / "confusion matrix"
    amplitude_dir = config.output_dir / "amplitude"
    save_amplitude_plots(train, synth_x, synth_y, class_names, amplitude_dir)

    print(f"Classes: {class_names}")
    print(f"Real train samples: {tuple(train['samples'].shape)}")
    print(f"Synthetic train samples: {tuple(synth_x.shape)} from {synthetic_path}")
    print(f"Testing both on: {tuple(test['samples'].shape)}")

    results = []
    models = parse_model_list(config.eval_models)
    for model in tqdm(models, desc="Models", dynamic_ncols=True):
        print(f"\n===== Evaluating model: {model} =====")
        config.model = model
        real_runs = []
        synthetic_runs = []
        for run_id in tqdm(range(config.num_eval), desc=f"{model} runs", leave=False, dynamic_ncols=True):
            real_runs.append(train_and_test(loaders["train"], loaders["test"], config, run_id, f"original/{model}", class_names, confusion_dir))
            synthetic_runs.append(train_and_test(synthetic_loader, loaders["test"], config, run_id, f"synthetic/{model}", class_names, confusion_dir))
        original = summarize(real_runs)
        synthetic = summarize(synthetic_runs)
        results.append({"model": model, "original": original, "synthetic": synthetic})

    report = {
        "class_names": class_names,
        "train_shape": list(train["samples"].shape),
        "synthetic_shape": list(synth_x.shape),
        "test_shape": list(test["samples"].shape),
        "num_eval": config.num_eval,
        "train_epochs": config.epoch_eval_train,
        "models": [item["model"] for item in results],
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    rows = table_rows(results)
    xlsx_path = table_base.with_suffix(".xlsx")
    csv_path = table_base.with_suffix(".csv")
    save_xlsx(xlsx_path, rows)
    save_csv(csv_path, rows)

    print("\nAccuracy comparison table")
    for row in rows:
        print(" | ".join(row))
    print(f"Saved report: {output_path}")
    print(f"Saved Excel table: {xlsx_path}")
    print(f"Saved CSV table: {csv_path}")
    print(f"Saved confusion matrices: {confusion_dir}")
    print(f"Saved amplitude plots: {amplitude_dir}")


if __name__ == "__main__":
    main()
