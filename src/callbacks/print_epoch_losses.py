# src/callbacks/print_epoch_losses.py
from lightning.pytorch.callbacks import Callback

class PrintEpochLosses(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics  
        tl = m.get("train/loss", None)
        vl = m.get("val/loss",   None)

        def fmt(x):
            try:
                return f"{(x.item() if hasattr(x, 'item') else float(x)):.4f}"
            except Exception:
                return "NA"

        print(f"[EPOCH {trainer.current_epoch}] train_loss={fmt(tl)} | val_loss={fmt(vl)}")

