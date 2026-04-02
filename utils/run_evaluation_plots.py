# utils/run_evaluation_plots.py
from __future__ import annotations
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

FPR_GRID = np.linspace(0.0, 1.0, 1001)

# ──────────────────────────────────────────────────────────────────────────────
# Inference collection
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_test_predictions(model, dataloader, device, encoder_mode="single"):
    """
    Run inference on test set, return (preds, labels, logits) as numpy arrays.
    Works for single, dual, and spectrum encoder modes.
    """
    model.eval()
    model.to(device)
    all_preds, all_labels, all_logits = [], [], []
    amp_enabled = device.type == "cuda"
    for batch in dataloader:
        labels_np = batch["labels"].cpu().numpy()
        batch_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            if encoder_mode == "spectrum":
                out = model(band_inputs=batch_device, labels=batch_device.get("labels"))
                logits = torch.exp(out["logits"])  # log-probs → probs
            elif encoder_mode == "dual":
                out = model(**batch_device)
                logits = torch.exp(out["logits"])  # log-probs → probs
            else:
                out = model(**batch_device)
                logits = out.logits if hasattr(out, "logits") else out["logits"]
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels_np)
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_logits)

# ──────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────
def _row_normalize_percent(cm_counts):
    cm = cm_counts.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    return norm * 100.0

def plot_confusion_matrix(cm_percent, class_names, outfile, title="", std_cm=None, dpi=350):
    """
    Plot a confusion matrix (row-normalized to %).
    If std_cm is provided, annotate cells as "XX.X% ± YY.Y%".
    """
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=dpi)
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    ax.set_xlabel("Predicted", fontsize=18)
    ax.set_ylabel("True", fontsize=18)
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, fontsize=14)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Accuracy (%)", fontsize=14)
    for i in range(n):
        for j in range(n):
            m = cm_percent[i, j]
            color = "black" if m < 60 else "white"
            if std_cm is not None:
                s = std_cm[i, j]
                txt = f"{m:.1f}%\n±{s:.1f}%"
            else:
                txt = f"{m:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12, color=color)
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    fig.tight_layout()
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix: {outfile}")

def save_per_run_confusion_matrix(preds, labels, class_names, outfile):
    """Compute and save a single-run confusion matrix figure."""
    n = len(class_names)
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    cm_pct = _row_normalize_percent(cm)
    plot_confusion_matrix(cm_pct, class_names, outfile)
    return cm_pct

def save_aggregated_confusion_matrix(all_cm_percents, class_names, outfile, title=""):
    """Save mean ± std confusion matrix across runs."""
    stacked = np.stack(all_cm_percents, axis=0)
    mean_cm = stacked.mean(axis=0)
    std_cm = stacked.std(axis=0)
    plot_confusion_matrix(mean_cm, class_names, outfile, title=title, std_cm=std_cm)

# ──────────────────────────────────────────────────────────────────────────────
# ROC Curves
# ──────────────────────────────────────────────────────────────────────────────
def _softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def _interp_on_grid(fpr, tpr, grid=FPR_GRID):
    if fpr[0] > 0.0:
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if fpr[-1] < 1.0:
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    return np.interp(grid, fpr, tpr)

def compute_roc_data(labels, logits, n_classes, scores_are_probs=False):
    """
    Compute macro and micro ROC curves + AUCs.
    Returns dict with 'macro_tpr', 'macro_auc', 'micro_tpr', 'micro_auc',
    'per_class' (list of dicts with 'tpr' and 'auc').
    """
    probs = logits if scores_are_probs else _softmax_np(logits)
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    per_class = []
    class_tprs = []
    for c in range(n_classes):
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], probs[:, c])
        tpr_g = _interp_on_grid(fpr_c, tpr_c)
        auc_c = auc(fpr_c, tpr_c)
        per_class.append({"tpr": tpr_g, "auc": float(auc_c)})
        class_tprs.append(tpr_g)
    class_tprs = np.stack(class_tprs, axis=0)
    macro_tpr = class_tprs.mean(axis=0)
    macro_auc = float(np.mean([pc["auc"] for pc in per_class]))
    fpr_mi, tpr_mi, _ = roc_curve(y_bin.ravel(), probs.ravel())
    micro_tpr = _interp_on_grid(fpr_mi, tpr_mi)
    micro_auc = float(auc(fpr_mi, tpr_mi))
    return {
        "macro_tpr": macro_tpr,
        "macro_auc": macro_auc,
        "micro_tpr": micro_tpr,
        "micro_auc": micro_auc,
        "per_class": per_class,
    }

def _apply_ticks(ax, major_step=0.1, minor_div=5, label_size=16, tick_size=10):
    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_div))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_div))
    ax.tick_params(labelsize=tick_size)
    ax.grid(axis="both", which="minor", alpha=0.15)
    ax.grid(axis="y", which="major", alpha=0.30)

def save_per_run_roc(roc_data, class_names, outfile, dpi=350):
    """Save single-run macro ROC figure."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=dpi)
    ax.plot(FPR_GRID, roc_data["macro_tpr"], linewidth=2, color="#1f77b4",
            label=f"Macro (AUC={roc_data['macro_auc']:.3f})")
    ax.plot(FPR_GRID, roc_data["micro_tpr"], linewidth=2, color="#ff7f0e",
            label=f"Micro (AUC={roc_data['micro_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="#888888")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    _apply_ticks(ax)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[INFO] Saved ROC curve: {outfile}")

def save_aggregated_roc(all_roc_data, class_names, outfile, title="", dpi=350):
    """
    Save aggregated ROC (mean ± std shaded band) for macro and micro.
    all_roc_data: list of dicts from compute_roc_data().
    """
    macro_tprs = np.stack([r["macro_tpr"] for r in all_roc_data], axis=0)
    micro_tprs = np.stack([r["micro_tpr"] for r in all_roc_data], axis=0)
    macro_aucs = [r["macro_auc"] for r in all_roc_data]
    micro_aucs = [r["micro_auc"] for r in all_roc_data]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=dpi)
    for arr, aucs, color, name in [
        (macro_tprs, macro_aucs, "#1f77b4", "Macro"),
        (micro_tprs, micro_aucs, "#ff7f0e", "Micro"),
    ]:
        m, s = arr.mean(axis=0), arr.std(axis=0)
        auc_m, auc_s = float(np.mean(aucs)), float(np.std(aucs))
        ax.plot(FPR_GRID, m, linewidth=2, color=color,
                label=f"{name} (AUC {auc_m:.3f}±{auc_s:.3f})")
        ax.fill_between(FPR_GRID, np.maximum(0, m - s), np.minimum(1, m + s),
                        alpha=0.15, linewidth=0, color=color)
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="#888888")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    _apply_ticks(ax)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[INFO] Saved aggregated ROC: {outfile}")

def save_aggregated_roc_per_class(all_roc_data, class_names, outdir, dpi=350):
    """Save per-class aggregated ROC (mean ± std) in separate files."""
    n_classes = len(class_names)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Also make a single figure with all classes
    fig_all, ax_all = plt.subplots(figsize=(7, 5), dpi=dpi)
    cmap = plt.cm.get_cmap("tab10", n_classes)
    for c in range(n_classes):
        tprs_c = np.stack([r["per_class"][c]["tpr"] for r in all_roc_data], axis=0)
        aucs_c = [r["per_class"][c]["auc"] for r in all_roc_data]
        m, s = tprs_c.mean(axis=0), tprs_c.std(axis=0)
        auc_m, auc_s = float(np.mean(aucs_c)), float(np.std(aucs_c))
        color = cmap(c)
        ax_all.plot(FPR_GRID, m, linewidth=2, color=color,
                    label=f"{class_names[c]} ({auc_m:.3f}±{auc_s:.3f})")
        ax_all.fill_between(FPR_GRID, np.maximum(0, m - s), np.minimum(1, m + s),
                            alpha=0.12, linewidth=0, color=color)
    ax_all.plot([0, 1], [0, 1], "--", linewidth=1, color="#888888")
    ax_all.set_xlim(0, 1); ax_all.set_ylim(0, 1)
    ax_all.set_xlabel("False Positive Rate", fontsize=16)
    ax_all.set_ylabel("True Positive Rate", fontsize=16)
    _apply_ticks(ax_all)
    ax_all.legend(loc="lower right", frameon=True, fontsize=9)
    fig_all.tight_layout()
    fig_all.savefig(outdir / "roc_per_class_aggregated.png", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_all)
    print(f"[INFO] Saved per-class aggregated ROC: {outdir / 'roc_per_class_aggregated.png'}")

# ──────────────────────────────────────────────────────────────────────────────
# Fusion Weights (Choquet) aggregated across runs
# ──────────────────────────────────────────────────────────────────────────────
def save_aggregated_fusion_weights(run_log_dirs, outfile, id2label=None, dpi=350):
    """
    Read fusion_weights.json from each run dir, compute mean ± std of g1/g2
    per epoch, and plot with shaded bands.
    run_log_dirs: list of Path to each run's logger directory (where fusion_weights.json lives).
    """
    g1_key = "g1_measure_singleton_source1(wave_if_p1>=p2)"
    g2_key = "g2_measure_singleton_source2(tf_if_p2>p1)"
    all_runs = []
    for rd in run_log_dirs:
        jp = Path(rd) / "fusion_weights.json"
        if not jp.exists():
            continue
        data = json.loads(jp.read_text())
        if not data:
            continue
        all_runs.append(data)
    if len(all_runs) < 2:
        print("[INFO] Not enough runs with fusion_weights.json for aggregated plot.")
        return
    # ---- Align by actual epoch numbers ----
    # Build per-run dict: epoch -> entry
    run_maps = []
    for run in all_runs:
        m = {}
        for entry in run:
            if "epoch" not in entry:
                continue
            m[int(entry["epoch"])] = entry
        if m:
            run_maps.append(m)
    if len(run_maps) < 2:
        print("[INFO] Not enough runs with valid epoch data for aggregated plot.")
        return
    # Use the UNION of epochs across runs, and at each epoch compute mean/std
    # over only the runs that have that epoch (no interpolation/padding).
    all_epochs = sorted(set().union(*(set(m.keys()) for m in run_maps)))
    if len(all_epochs) < 2:
        print("[INFO] Not enough epoch points across runs for aggregated plot.")
        return
    # Determine whether per-class from the first available epoch in the first run
    first_run = run_maps[0]
    first_ep = sorted(first_run.keys())[0]
    sample_val = first_run[first_ep].get(g1_key, None)
    if sample_val is None:
        print(f"[INFO] Missing key '{g1_key}' in fusion_weights.json entries; cannot aggregate.")
        return
    is_per_class = isinstance(sample_val, list)
    epochs = all_epochs
    # Helper: compute mean/std at each epoch across available runs
    def _stats_at_epoch(ep, key):
        vals = []
        for m in run_maps:
            if ep in m and key in m[ep]:
                vals.append(m[ep][key])
        if len(vals) == 0:
            return None, None  
        arr = np.array(vals, dtype=np.float64)
        return arr.mean(axis=0), arr.std(axis=0)
    # ------------------------------------------------------------------------
    if is_per_class:
        n_classes = len(sample_val)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
        cmap = plt.cm.get_cmap("tab10", n_classes)
        for c in range(n_classes):
            label = id2label.get(c, f"Class {c}") if id2label else f"Class {c}"
            color = cmap(c)
            g1_m, g1_s = [], []
            g2_m, g2_s = [], []
            for ep in epochs:
                m1, s1 = _stats_at_epoch(ep, g1_key)
                m2, s2 = _stats_at_epoch(ep, g2_key)
                # m1/m2 are vectors (n_classes,)
                g1_m.append(m1[c] if m1 is not None else np.nan)
                g1_s.append(s1[c] if s1 is not None else np.nan)
                g2_m.append(m2[c] if m2 is not None else np.nan)
                g2_s.append(s2[c] if s2 is not None else np.nan)
            g1_m = np.array(g1_m, dtype=np.float64)
            g1_s = np.array(g1_s, dtype=np.float64)
            g2_m = np.array(g2_m, dtype=np.float64)
            g2_s = np.array(g2_s, dtype=np.float64)
            axes[0].plot(epochs, g1_m, linewidth=2, color=color, label=label)
            axes[0].fill_between(
                epochs,
                np.maximum(0, g1_m - g1_s),
                np.minimum(1, g1_m + g1_s),
                alpha=0.15, linewidth=0, color=color
            )
            axes[1].plot(epochs, g2_m, linewidth=2, color=color, label=label)
            axes[1].fill_between(
                epochs,
                np.maximum(0, g2_m - g2_s),
                np.minimum(1, g2_m + g2_s),
                alpha=0.15, linewidth=0, color=color
            )
        for ax in axes:
            ax.set_xlabel("Epoch", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.set_xlim(min(epochs), max(epochs))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9, framealpha=0.8)
            ax.tick_params(labelsize=12)
        axes[0].set_ylabel("g1 (wave preference)", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("g2 (TF preference)", fontsize=14, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
        g1_m, g1_s = [], []
        g2_m, g2_s = [], []
        for ep in epochs:
            m1, s1 = _stats_at_epoch(ep, g1_key)
            m2, s2 = _stats_at_epoch(ep, g2_key)
            g1_m.append(float(m1) if m1 is not None else np.nan)
            g1_s.append(float(s1) if s1 is not None else np.nan)
            g2_m.append(float(m2) if m2 is not None else np.nan)
            g2_s.append(float(s2) if s2 is not None else np.nan)
        g1_m = np.array(g1_m, dtype=np.float64)
        g1_s = np.array(g1_s, dtype=np.float64)
        g2_m = np.array(g2_m, dtype=np.float64)
        g2_s = np.array(g2_s, dtype=np.float64)
        for m, s, color, label in [
            (g1_m, g1_s, "#1f77b4", "g1 (wave preference)"),
            (g2_m, g2_s, "#ff7f0e", "g2 (TF preference)"),
        ]:
            ax.plot(epochs, m, linewidth=2.5, color=color, label=label)
            ax.fill_between(
                epochs,
                np.maximum(0, m - s),
                np.minimum(1, m + s),
                alpha=0.15, linewidth=0, color=color
            )
        ax.set_xlabel("Epoch", fontsize=14, fontweight="bold")
        ax.set_ylabel("Fuzzy Measure Weight", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlim(min(epochs), max(epochs))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=12)
        ax.tick_params(labelsize=12)
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[INFO] Saved aggregated fusion weights: {outfile}")

# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (for aggregated_metrics.json)
# ──────────────────────────────────────────────────────────────────────────────
def compute_macro_auc_from_labels_logits(labels: np.ndarray, logits: np.ndarray, n_classes: int) -> float:
    """
    Convenience helper: return scalar macro AUC (mean of per-class one-vs-rest AUCs).
    Uses the same softmax + label_binarize convention as compute_roc_data().
    """
    roc_data = compute_roc_data(labels, logits, n_classes, scores_are_probs=True)
    return float(roc_data.get("macro_auc", 0.0))