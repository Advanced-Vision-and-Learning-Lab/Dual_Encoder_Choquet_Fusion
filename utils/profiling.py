# utils/profiling.py
import time
import contextlib
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn


def _autocast_if_cuda(device, enabled: bool = False, dtype=torch.float16):
    # Default disabled to avoid dtype mismatches (robust).
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _first_label_key(batch: Dict[str, Any]) -> str:
    for k in ("labels", "label", "y", "targets", "target"):
        if k in batch:
            return k
    raise KeyError(f"No label key found in batch. Keys present: {list(batch.keys())}")


def _to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def _match_batch_float_dtype(batch, dtype):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(dtype=dtype)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _slice_batch_to_bsz1(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied batch where leading-batch-dim tensors are sliced to bsz=1."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) >= 1:
            out[k] = v[:1]
        else:
            out[k] = v
    return out


@torch.no_grad()
def profile_inference(model, batch, device, repeats: int = 50, warmup: int = 10, use_mixed_precision: bool = False) -> Dict[str, Any]:
    """
    Forward-only timing (eval mode). Standard inference-time procedure:
      - warmup iterations
      - CUDA synchronize and event timing for accuracy
      - per-sample latency (mean/std), throughput (samples/s), and peak alloc/reserved MB

    Returns:
      - infer_ms_per_sample_mean / infer_ms_per_sample_std
      - infer_throughput_sps_mean / infer_throughput_sps_std
      - infer_peak_alloc_mb / infer_peak_reserved_mb
    """
    model.eval()
    batch = _to_device(batch, device)
    model_dtype = next(model.parameters()).dtype
    batch = _match_batch_float_dtype(batch, model_dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup
    with torch.inference_mode(), _autocast_if_cuda(device, enabled=use_mixed_precision):
        for _ in range(warmup):
            _ = model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed loop
    times = []
    with torch.inference_mode(), _autocast_if_cuda(device, enabled=use_mixed_precision):
        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(True); end = torch.cuda.Event(True)
                start.record()
                _ = model(**batch)
                end.record(); torch.cuda.synchronize()
                times.append(start.elapsed_time(end))  # ms for whole batch
            else:
                t0 = time.time()
                _ = model(**batch)
                times.append((time.time() - t0) * 1000.0)

    times = np.asarray(times, dtype=np.float64)
    first_tensor = next(v for v in batch.values() if torch.is_tensor(v))
    bsz = int(first_tensor.size(0))

    # Latency (ms/sample)
    mean_ms_per_sample = float(times.mean() / bsz)
    std_ms_per_sample  = float(times.std()  / bsz)

    # Throughput (samples/s)
    throughputs = bsz / (times / 1000.0)
    thr_mean = float(throughputs.mean())
    thr_std  = float(throughputs.std())

    # Peak memory
    peak_alloc = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else None
    peak_resvd = float(torch.cuda.max_memory_reserved()  / (1024**2)) if device.type == "cuda" else None

    return {
        "infer_ms_per_sample_mean":  mean_ms_per_sample,
        "infer_ms_per_sample_std":   std_ms_per_sample,
        "infer_throughput_sps_mean": thr_mean,
        "infer_throughput_sps_std":  thr_std,
        "infer_peak_alloc_mb":       peak_alloc,
        "infer_peak_reserved_mb":    peak_resvd,
    }

def profile_train_step(
    model,
    batch,
    device,
    lr: float,
    warmup_steps: int = 3,
    repeats: int = 3,
    use_mixed_precision: bool = False,
):
    """
    One full TRAIN step probe (fwd + loss + bwd + optimizer.step).
    """
    model.train()
    batch = _to_device(batch, device)
    model_dtype = next(model.parameters()).dtype
    batch = _match_batch_float_dtype(batch, model_dtype)
    label_key = _first_label_key(batch)

    # Save weights so the probe doesn't change initialization
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Optimizer over *trainable* params only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda" and use_mixed_precision))
    loss_fn = torch.nn.CrossEntropyLoss()

    def _extract_logits(out):
        """Robust logits extraction from various output formats."""
        if hasattr(out, "logits"):
            return out.logits
        elif isinstance(out, dict) and "logits" in out:
            return out["logits"]
        elif isinstance(out, (tuple, list)):
            return out[0]
        else:
            return out

    # Warmup (allocate optimizer states, build graph)
    for _ in range(warmup_steps):
        opt.zero_grad(set_to_none=True)
        with _autocast_if_cuda(device, enabled=use_mixed_precision):
            out = model(**batch)
            logits = _extract_logits(out)
            loss = loss_fn(logits, batch[label_key])
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed steps
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
        t0 = time.time()

        opt.zero_grad(set_to_none=True)
        with _autocast_if_cuda(device, enabled=use_mixed_precision):
            out = model(**batch)
            logits = _extract_logits(out)  # ✅ Use helper function here too
            loss = loss_fn(logits, batch[label_key])
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            times.append((time.time() - t0) * 1000.0)

    # Peak memory during the train step
    peak_alloc = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else None
    peak_resvd = float(torch.cuda.max_memory_reserved() / (1024**2)) if device.type == "cuda" else None

    # Restore weights; clear grads
    model.load_state_dict(state_cpu, strict=True)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stats
    times = np.asarray(times, dtype=np.float64)
    bsz = int(batch[label_key].size(0))

    ms_per_batch_mean = float(times.mean())
    ms_per_batch_std = float(times.std())
    ms_per_sample_mean = float(ms_per_batch_mean / bsz)
    ms_per_sample_std = float(ms_per_batch_std / bsz)

    throughputs = bsz / (times / 1000.0)
    thr_mean = float(throughputs.mean())
    thr_std = float(throughputs.std())

    return {
        "train_ms_per_batch_mean": ms_per_batch_mean,
        "train_ms_per_batch_std": ms_per_batch_std,
        "train_ms_per_sample_mean": ms_per_sample_mean,
        "train_ms_per_sample_std": ms_per_sample_std,
        "train_throughput_sps_mean": thr_mean,
        "train_throughput_sps_std": thr_std,
        "train_peak_alloc_mb": peak_alloc,
        "train_peak_reserved_mb": peak_resvd,
    }

@torch.no_grad()
def profile_flops_bsz1(model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manual FLOPs for a single forward (bsz=1) on real inputs.
    Counts common ops with 2 FLOPs per MAC:
      - Linear: 2 * in_features * out_features * (#vectors)
      - Conv1d: 2 * B * Cout * Lout * (Cin/groups * K)
      - Conv2d: 2 * B * Cout * Hout * Wout * (Cin/groups * Kh * Kw)
      - Attention core (excl. qkv/out linears): 4 * B * H * N * N * d
        where input to the attention module is (B, N, C), H = num_heads, d = C/H
    Returns: {"flops_bsz1": int|None, "gflops_bsz1": float|None}
    """
    # 1) Build a bsz=1, label-free input; match device/dtype to model
    b1 = _slice_batch_to_bsz1(batch)
    # drop label-like keys
    for key in ("labels", "label", "y", "targets", "target"):
        if key in b1:
            b1.pop(key)
    try:
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
    except StopIteration:
        dev, dt = torch.device("cpu"), torch.float32
    for k, v in list(b1.items()):
        if torch.is_tensor(v):
            b1[k] = v.to(device=dev, dtype=(dt if v.is_floating_point() else v.dtype))

    model.eval()

    total = 0
    handles = []

    def add(v):
        nonlocal total
        if isinstance(v, (int, float)):
            total += int(v)

    def _safe_shape(x: torch.Tensor):
        # Returns tuple of ints for shape; avoids tracing issues.
        return tuple(int(s) for s in x.shape)

    # ---- Linear ----
    def linear_hook(mod: nn.Linear, inp, out):
        try:
            x = inp[0]
            if not torch.is_tensor(x) or x.dim() == 0:
                return
            in_f = int(mod.in_features)
            out_f = int(mod.out_features)
            # Expect last dim == in_features; number of vectors is product of all prev dims
            if x.shape[-1] != in_f:
                # if it's mismatched, bail quietly (avoid wrong counts)
                return
            vecs = int(np.prod(_safe_shape(x)[:-1])) if x.dim() > 1 else 1
            add(2 * vecs * in_f * out_f)
        except Exception:
            return

    # ---- Conv2d ----
    def conv2d_hook(mod: nn.Conv2d, inp, out):
        try:
            y = out if torch.is_tensor(out) else None
            if y is None or y.dim() != 4:
                return
            B, Cout, H, W = _safe_shape(y)
            Cin = int(mod.in_channels)
            if isinstance(mod.kernel_size, tuple):
                Kh, Kw = int(mod.kernel_size[0]), int(mod.kernel_size[1])
            else:
                Kh = Kw = int(mod.kernel_size)
            groups = int(mod.groups)
            macs_per_out = (Cin // groups) * Kh * Kw
            add(2 * B * Cout * H * W * macs_per_out)
        except Exception:
            return

    # ---- Conv1d ----
    def conv1d_hook(mod: nn.Conv1d, inp, out):
        try:
            y = out if torch.is_tensor(out) else None
            if y is None or y.dim() != 3:
                return
            B, Cout, L = _safe_shape(y)
            Cin = int(mod.in_channels)
            K = int(mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size)
            groups = int(mod.groups)
            macs_per_out = (Cin // groups) * K
            add(2 * B * Cout * L * macs_per_out)
        except Exception:
            return

    # ---- Attention core (QK^T + softmaxV) ----
    def attn_pre_hook(mod, args, kwargs):
        try:
            H = getattr(mod, "num_heads", None)
            if H is None:
                H = getattr(mod, "num_attention_heads", None)
            if not isinstance(H, int) or H <= 0:
                return
            x = kwargs.get("hidden_states", None)
            if x is None:
                for a in args:
                    if torch.is_tensor(a):
                        x = a; break
            if x is None or not torch.is_tensor(x) or x.dim() != 3:
                return
            B, N, C = (int(s) for s in x.shape)
            if C % H != 0:
                return
            d = C // H
            add(4 * B * H * N * N * d)
        except Exception:
            return

    # Register hooks
    for m in model.modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, nn.Conv1d):
            handles.append(m.register_forward_hook(conv1d_hook))
        else:
            if hasattr(m, "num_heads") or hasattr(m, "num_attention_heads"):
                handles.append(m.register_forward_pre_hook(attn_pre_hook, with_kwargs=True))

    # Single forward pass; inference mode avoids grad/state changes
    try:
        with torch.inference_mode():
            _ = model(**b1)
    finally:
        for h in handles:
            h.remove()

    if total > 0:
        return {"flops_bsz1": int(total), "gflops_bsz1": float(total) / 1e9}
    return {"flops_bsz1": None, "gflops_bsz1": None}
