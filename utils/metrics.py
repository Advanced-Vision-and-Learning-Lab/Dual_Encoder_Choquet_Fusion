# utils/metrics.py
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def _mean(x: List[float]):
    return float(np.mean(x)) if x else None

def _std(x: List[float]):
    return float(np.std(x)) if x else None

def aggregate_and_save(all_metrics: List[Dict[str, Any]], exp_dir: Path) -> None:
    # --- Accuracies ---
    # Prefer explicit 'val_accuracy' if present; otherwise fall back to 'accuracy'
    val_list = []
    for m in all_metrics:
        v = m.get("val_accuracy")
        if v is None:
            v = m.get("accuracy")
        if v is not None:
            val_list.append(v)

    test_list = [m.get("test_accuracy") for m in all_metrics if m.get("test_accuracy") is not None]

    # --- New: test F1 + macro AUC ---
    test_f1_list = [m.get("test_f1") for m in all_metrics if m.get("test_f1") is not None]
    test_macro_auc_list = [m.get("test_macro_auc") for m in all_metrics if m.get("test_macro_auc") is not None]

    # --- Compute metrics ---
    gflops = [m.get("gflops_bsz1") for m in all_metrics if m.get("gflops_bsz1") is not None]
    
    # Memory: check both train_peak_alloc_mb and infer_peak_alloc_mb
    mem_mb = []
    for m in all_metrics:
        v = m.get("train_peak_alloc_mb") or m.get("infer_peak_alloc_mb")
        if v is not None:
            mem_mb.append(v)
    
    # Latency
    lat_ms = [m.get("latency_ms_per_sample") for m in all_metrics if m.get("latency_ms_per_sample") is not None]
    
    # Throughput: prefer train, fall back to infer
    thr_sps = []
    for m in all_metrics:
        v = m.get("train_throughput_sps_mean") or m.get("infer_throughput_sps_mean")
        if v is not None:
            thr_sps.append(v)

    # --- Params ---
    total_params = all_metrics[0].get("total_params") if all_metrics else None
    train_params = all_metrics[0].get("trainable_params") if all_metrics else None
    if total_params is not None and train_params is not None:
        train_pct = 100.0 * float(train_params) / float(total_params) if total_params else 0.0
    else:
        train_pct = None

    # --- average trainable params across runs ---
    trainable_vals = [m.get("trainable_params") for m in all_metrics if m.get("trainable_params") is not None]
    avg_trainable_params = float(np.mean(trainable_vals)) if trainable_vals else None

    aggregated = {
        # Accuracies
        "val_accuracy_mean": _mean(val_list),
        "val_accuracy_std": _std(val_list),
        "test_accuracy_mean": _mean(test_list),
        "test_accuracy_std": _std(test_list),

        # New: F1 + macro AUC
        "test_f1_mean": _mean(test_f1_list),
        "test_f1_std": _std(test_f1_list),
        "test_macro_auc_mean": _mean(test_macro_auc_list),
        "test_macro_auc_std": _std(test_macro_auc_list),

        # Four compute metrics
        "gflops_bsz1_mean": _mean(gflops),
        "gflops_bsz1_std": _std(gflops),
        "train_peak_alloc_mb_mean": _mean(mem_mb),
        "train_peak_alloc_mb_std": _std(mem_mb),
        "latency_ms_per_sample_mean": _mean(lat_ms),
        "latency_ms_per_sample_std": _std(lat_ms),
        "train_throughput_sps_mean": _mean(thr_sps),
        "train_throughput_sps_std": _std(thr_sps),

        # Params
        "total_params": int(total_params) if total_params is not None else None,
        "trainable_params": int(train_params) if train_params is not None else None,
        "trainable_params_pct": float(train_pct) if train_pct is not None else None,
        "avg_trainable_params": avg_trainable_params,
    }

    agg_path = Path(exp_dir) / "aggregated_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)