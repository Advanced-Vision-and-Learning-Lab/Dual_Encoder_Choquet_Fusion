# src/callbacks/hpt_gate_logger.py
from __future__ import annotations
from pathlib import Path
import json
import torch
import lightning as L
from typing import Dict, List, Set
from src.hpt import HistogramAdapterParallel

def _read_gate(m: HistogramAdapterParallel) -> float:
    with torch.no_grad():
        return float(m.gate.item())

class HPTGateLogger(L.Callback):
    def __init__(self, filename: str = "hpt_gates.json"):
        super().__init__()
        self.filename = filename
        self._seen_ptrs: Set[int] = set()

    def _run_dir(self, trainer: L.Trainer) -> Path:
        log_dir = getattr(trainer.logger, "log_dir", None)
        return Path(log_dir) if log_dir is not None else Path("lightning_logs")

    def _collect(self, pl_module: L.LightningModule) -> List[Dict]:
        rows: List[Dict] = []
        self._seen_ptrs.clear()
        for name, mod in pl_module.model.named_modules():
            if isinstance(mod, HistogramAdapterParallel):
                pid = id(mod)
                if pid in self._seen_ptrs:
                    continue
                self._seen_ptrs.add(pid)
                rows.append({"module": name, "gate": _read_gate(mod)})
        return rows

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        run_dir = self._run_dir(trainer); run_dir.mkdir(parents=True, exist_ok=True)
        fpath = run_dir / self.filename
        epoch = int(trainer.current_epoch)
        rows = self._collect(pl_module)
        for r in rows:
            r["epoch"] = epoch
        data = []
        if fpath.exists():
            with fpath.open("r") as f:
                data = json.load(f)
        data.extend(rows)
        with fpath.open("w") as f:
            json.dump(data, f, indent=2)
