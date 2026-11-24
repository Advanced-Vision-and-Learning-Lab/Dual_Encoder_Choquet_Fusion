# src/model_module.py
from typing import Dict, Optional
import inspect
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import AutoConfig, AutoModelForAudioClassification, AutoModelForImageClassification
from peft import LoraConfig, IA3Config, get_peft_model, TaskType
from src.adapters import apply_adapter
from src.hpt import apply_hpt
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint

def _freeze_classifier_layernorm(model: torch.nn.Module) -> None:
    """Freeze any nn.LayerNorm inside model.classifier."""
    clf = getattr(model, "classifier", None)
    if clf is None:
        return
    for m in clf.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = False

def _unfreeze_last_classifier_linear(model: torch.nn.Module) -> None:
    """Unfreeze ONLY the last nn.Linear inside model.classifier."""
    clf = getattr(model, "classifier", None)
    if clf is None:
        return
    linears = [m for m in clf.modules() if isinstance(m, torch.nn.Linear)]
    if linears:
        for p in linears[-1].parameters():
            p.requires_grad = True

def _infer_role_from_name(name: str) -> str:
    n = name.lower()
    if "q_proj" in n or ".query" in n or "query" in n:
        return "q"
    if "k_proj" in n or ".key" in n or "key" in n:
        return "k"
    if "v_proj" in n or ".value" in n or "value" in n:
        return "v"
    # FFN indicators across HuggingFace / Wav2Vec2 / AST
    if ("intermediate" in n or "ffn" in n or "output.dense" in n 
        or ".mlp." in n or "feedforward" in n):
        return "ffn"
    return "other"

def tie_lora_adapters(model: torch.nn.Module, adapter_name: str = "default"):
    """
    Make all LoRA A/B weights shared within each (role, A_shape, B_shape) bucket.
      - module.lora_A[adapter_name].weight  (nn.Parameter)
      - module.lora_B[adapter_name].weight  (nn.Parameter)
    """
    name_to_mod = dict(model.named_modules())
    # (role, A_shape, B_shape) -> [(mod, A, B)]
    buckets = {}
    for mod_name, mod in name_to_mod.items():
        lora_A = getattr(mod, "lora_A", None)
        lora_B = getattr(mod, "lora_B", None)
        if lora_A is None or lora_B is None:
            continue
        if adapter_name not in getattr(lora_A, "_modules", {}):
            continue
        if adapter_name not in getattr(lora_B, "_modules", {}):
            continue
        A = lora_A._modules[adapter_name]  # nn.Linear
        B = lora_B._modules[adapter_name]  # nn.Linear
        if not (hasattr(A, "weight") and isinstance(A.weight, torch.nn.Parameter)): 
            continue
        if not (hasattr(B, "weight") and isinstance(B.weight, torch.nn.Parameter)):
            continue
        role = _infer_role_from_name(mod_name)
        key = (role, tuple(A.weight.shape), tuple(B.weight.shape))
        buckets.setdefault(key, []).append((mod, A, B))
    for key, items in buckets.items():
        if len(items) < 2:
            continue
        _, A_master, B_master = items[0]
        Aw = A_master.weight
        Bw = B_master.weight
        for (_, A_i, B_i) in items[1:]:
            if tuple(A_i.weight.shape) == tuple(Aw.shape):
                A_i.weight = Aw
            if tuple(B_i.weight.shape) == tuple(Bw.shape):
                B_i.weight = Bw

def tie_ia3_vectors(model: torch.nn.Module, adapter_name: str = "default", mode: str = "kv_ffn"):
    """
    Share IA3 scaling vectors among modules of the same role (k, v, ffn) with identical shapes.
    mode:
      - "v_only": share only V projections
      - "kv_ffn": share K, V, and FFN
    """
    name_to_mod = dict(model.named_modules())
    # (role, shape) -> [(container, param_key, mod_name)]
    buckets = {}
    for mod_name, mod in name_to_mod.items():
        ia3_l = getattr(mod, "ia3_l", None)
        if ia3_l is None:
            continue
        param = None
        if hasattr(ia3_l, "_parameters") and adapter_name in ia3_l._parameters:
            param = ia3_l._parameters[adapter_name]
            container = ia3_l
            param_key = adapter_name
        else:
            continue
        if not isinstance(param, torch.nn.Parameter):
            continue
        role = _infer_role_from_name(mod_name)
        if mode == "v_only" and role not in ("v",):
            continue
        if mode == "kv_ffn" and role not in ("k", "v", "ffn"):
            continue
        key = (role, tuple(param.shape))
        buckets.setdefault(key, []).append((container, param_key, mod_name))
    for key, items in buckets.items():
        if len(items) < 2:
            continue
        cont_master, pname_master, _ = items[0]
        master_param = cont_master._parameters[pname_master]
        for (cont_i, pname_i, _) in items[1:]:
            cont_i._parameters[pname_i] = master_param

class AudioClassifierModule(L.LightningModule):
    """
    LightningModule wrapping a pretrained audio classification model.
    - Loads model and configures label mappings.
    - Applies PEFT method if requested, freezing other parameters.
    """
    @staticmethod
    def _freeze_wavlike_feature_extractor(m):
        """
        Freeze only the raw conv feature_extractor (feature encoder) for
        wav2vec2 / hubert / wavlm backbones.
        """
        # Prefer official APIs
        if hasattr(m, "freeze_feature_encoder") and callable(getattr(m, "freeze_feature_encoder")):
            m.freeze_feature_encoder()
            return
        if hasattr(m, "freeze_feature_extractor") and callable(getattr(m, "freeze_feature_extractor")):
            m.freeze_feature_extractor()
            return
        # locate and freeze just the feature_extractor
        for parent_attr in ("wav2vec2", "hubert", "wavlm"):
            if hasattr(m, parent_attr):
                parent = getattr(m, parent_attr)
                fe = getattr(parent, "feature_extractor", None)
                if fe is not None:
                    for p in fe.parameters():
                        p.requires_grad = False
                break

    def __init__(
        self,
        model_name: str,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        peft_method: str = "none",
        freeze_fe: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target: str = "all",
        adapter_dim: int = 64,
        adapter_dropout: float = 0.0,
        adapter_insertion: str = "ffn",
        adapter_shared: bool = False,
        hpt_bins: int = 16,
        hpt_location: str = "mhsa",       
        hpt_shared: bool = False,
        hpt_init_rand: bool = False,
        ia3_target: str = "kv_ffn",
        lora_shared: bool = False,   
        ia3_shared: bool = False,    
        lr: float = 5e-4,
        ast_max_length: int = 512,
        hpt_adaptive: bool = False,
        dataset: str = "",
        ast_input_type: str = "logmel",
        mfcc_n_mfcc: int = 32,
        cqt_n_bins: int = 84,
        init_rand: Optional[bool] = None,
    ):
        super().__init__()
        init_rand = hpt_init_rand if init_rand is None else init_rand
        self.save_hyperparameters()
        self._dataset = dataset
        self._backbone_model_type = None

        # Configure model settings
        config = AutoConfig.from_pretrained(model_name)
        self._backbone_model_type = config.model_type
        config.num_labels = len(id2label)
        config.id2label = id2label
        config.label2id = label2id

        # Adjust spectrogram parameters if AST
        if config.model_type == "audio-spectrogram-transformer":
            if ast_input_type == "mfcc":
                config.num_mel_bins = mfcc_n_mfcc
            elif ast_input_type in ["cqt", "vqt"]:
                config.num_mel_bins = cqt_n_bins 
            else:
                config.num_mel_bins = 64

            config.max_length = ast_max_length

        # Load pretrained model (image vs audio)
        vision_model_types = {"vit"}
        try:
            model_type = config.model_type
        except Exception:
            model_type = None
        if model_type in vision_model_types:
            base_model = AutoModelForImageClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            base_model = AutoModelForAudioClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )

        # ── restore pretrained AST position embeddings via interpolation ──
        if config.model_type == "audio-spectrogram-transformer":
            import safetensors.torch as st
            from transformers.utils import cached_file
            # Load only the position embedding tensor from the pretrained checkpoint
            ckpt_path = cached_file(model_name, "model.safetensors")
            key = "audio_spectrogram_transformer.embeddings.position_embeddings"
            old_pos = st.load_file(ckpt_path)[key]  # [1, 1214, 768]
            new_pos_param = base_model.audio_spectrogram_transformer.embeddings.position_embeddings
            new_len = new_pos_param.shape[1]
            old_len = old_pos.shape[1]
            if new_len != old_len:
                old_pos_t = old_pos.permute(0, 2, 1)  # [1, 768, old_len]
                new_pos = F.interpolate(old_pos_t, size=new_len, mode="linear", align_corners=False)
                new_pos = new_pos.permute(0, 2, 1)    # [1, new_len, 768]
                new_pos_param.data.copy_(new_pos)
                print(f"[INFO] Interpolated AST position embeddings: [{old_len}] → [{new_len}]")

        # ---- collapse AVES/HuBERT projector + classifier to 768 -> num_labels ----
        if config.model_type == "hubert":
            hidden_size = getattr(config, "hidden_size", 768)
            num_labels = config.num_labels
            # Remove the 768 -> 256 projector: make it identity
            if hasattr(base_model, "projector"):
                base_model.projector = nn.Identity()
            # Replace the classifier with a direct 768 -> num_labels layer
            base_model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_labels)
            )

        base_model.gradient_checkpointing_disable()

        # freeze raw feature extractor for wav2vec2/Hubert
        if freeze_fe and config.model_type in ("wav2vec2", "hubert", "wavlm"):
            self._freeze_wavlike_feature_extractor(base_model)

        # Wrap forward to filter unsupported kwargs and enable pos encoding
        allowed_args = set(inspect.signature(base_model.forward).parameters)
        def safe_forward(self_, *args, **kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in allowed_args}
            if "interpolate_pos_encoding" in allowed_args:
                filtered["interpolate_pos_encoding"] = True
            return base_model.__class__.forward(base_model, *args, **filtered)
        base_model.forward = safe_forward.__get__(base_model, base_model.__class__)

        # Apply PEFT method
        method = peft_method.lower()
        if method == "lora":

            def discover_target_modules(model, target: str):
                """
                Return a list of substrings that PEFT will match against module names.
                - ViT/AST use:  'query', 'key', 'value'
                - wav2vec2-family use: 'q_proj', 'k_proj', 'v_proj'
                """
                # Map generic roles to tokens used across backbones
                TOKENS = {
                    "query": ("query", "q_proj"),
                    "key":   ("key",   "k_proj"),
                    "value": ("value", "v_proj"),
                }
                if target == "all":
                    return "all-linear"
                if target == "qv":
                    present = set()
                    for name, _ in model.named_modules():
                        for tok in TOKENS["query"]:
                            if tok in name:
                                present.add(tok)
                        for tok in TOKENS["value"]:
                            if tok in name:
                                present.add(tok)
                    if not present:
                        raise ValueError("LoRA target 'qv' matched no modules in this backbone.")
                    return sorted(present)
                if target == "query":
                    present = sorted({tok for name, _ in model.named_modules() for tok in TOKENS["query"] if tok in name})
                    if not present:
                        raise ValueError("LoRA target 'query' matched no modules in this backbone.")
                    return present
                raise ValueError(f"Unsupported lora_target='{target}'.")
            target_modules = discover_target_modules(base_model, lora_target)
            if isinstance(target_modules, list) and len(target_modules) == 0:
                raise ValueError(f"No LoRA targets found with lora_target='{lora_target}'.")
            print(f"[LoRA] Using target_modules={target_modules}")
            # Instantiate the LoRA‑wrapped model
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            self.model = get_peft_model(base_model, lora_cfg)
            if self.hparams.lora_shared:
                tie_lora_adapters(self.model, adapter_name="default")
            # Freeze everything except the LoRA adapters and the final classifier head
            for name, param in self.model.named_parameters():
                if not (
                    "lora_" in name or
                    "modules_to_save" in name or
                    "classifier" in name
                ):
                    param.requires_grad = False
            _freeze_classifier_layernorm(self.model)        
        elif method == "ia3":
            def discover_ia3_targets(model, mode="kv_ffn"):
                attn_k, attn_v, ffn = set(), set(), set()
                for name, module in model.named_modules():
                    if not isinstance(module, nn.Linear):
                        continue
                    n = name.lower()
                    # attention K/V
                    if ("k_proj" in n) or (".key" in n) or (n.endswith(".key")):
                        attn_k.add("k_proj" if "k_proj" in n else "key")
                    if ("v_proj" in n) or (".value" in n) or (n.endswith(".value")):
                        attn_v.add("v_proj" if "v_proj" in n else "value")
                    # FFN indicators across HF/Wav2Vec2/AST/ViT-like stacks
                    if ("intermediate" in n or "ffn" in n or "output.dense" in n
                        or ".mlp." in n or "feedforward" in n):
                        ffn.add(name)  # use the module's full name
                    if "intermediate_dense" in n: ffn.add("intermediate_dense")
                    if "intermediate.dense" in n: ffn.add("intermediate.dense")
                if mode == "v_only":
                    target_modules = sorted(attn_v)
                    feedforward_modules = []
                elif mode == "kv_ffn":
                    target_modules = sorted(attn_k | attn_v | ffn)
                    feedforward_modules = sorted(ffn)
                else:
                    raise ValueError(f"Unknown ia3_target mode: {mode}")
                return target_modules, feedforward_modules
            tmods, fmods = discover_ia3_targets(base_model, self.hparams.ia3_target)
            ia3_cfg = IA3Config(
                task_type=TaskType.SEQ_CLS,
                target_modules=tmods,
                feedforward_modules=fmods,
                inference_mode=False,
            )
            self.model = get_peft_model(base_model, ia3_cfg)
            if self.hparams.ia3_shared:
                tie_ia3_vectors(self.model, adapter_name="default", mode=self.hparams.ia3_target)
            # Freeze everything except IA3 params + classifier head 
            for name, param in self.model.named_parameters():
                if not ("ia3" in name.lower() or "modules_to_save" in name or "classifier" in name.lower()):
                    param.requires_grad = False
            _freeze_classifier_layernorm(self.model)
        elif method == "adapter":
            self.model = apply_adapter(
                base_model,
                adapter_dim=adapter_dim,
                adapter_dropout=adapter_dropout,
                insertion=adapter_insertion,
                shared=adapter_shared)
            
            _freeze_classifier_layernorm(self.model)
            _unfreeze_last_classifier_linear(self.model)
        elif method == "hpt":
            self.model, _ = apply_hpt(
                base_model,
                num_bins=hpt_bins,
                location=hpt_location,
                shared=hpt_shared,
                init_rand=init_rand,
            )
            _freeze_classifier_layernorm(self.model)
            _unfreeze_last_classifier_linear(self.model)
            print(f"\nHPT attached: location={hpt_location}, shared={hpt_shared}, bins={hpt_bins}")
        elif method == "linear":
            # strict linear probe: freeze everything, train only the last Linear in classifier
            self.model = base_model
            for p in self.model.parameters():
                p.requires_grad = False
            _unfreeze_last_classifier_linear(self.model)  # train final Linear
            _freeze_classifier_layernorm(self.model)      # keep LN in head frozen
        else:
            # Full fine-tuning
            self.model = base_model

        # Bind forward implementation once 
        if method == "linear":
            def _forward_impl(**batch):
                was_training = self.model.training
                self.model.eval()
                try:
                    return self.model(**batch)
                finally:
                    if was_training:
                        self.model.train()
        else:
            def _forward_impl(**batch):
                return self.model(**batch)
        self._forward_impl = _forward_impl
        self.best_epoch = -1
        self.best_val_acc = -1.0
        self._first_validation_run_completed = False
        # ── Loss history for learning-curve plot ──
        self._loss_history = {"train": [], "val": []}
        # Report parameter counts
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        percent = 100 * trainable / total
        print(f"\nModel initialized with PEFT method: {method.upper()}")
        print(f"Trainable parameters: {trainable:,} / {total:,} ({percent:.2f}%)")
        if freeze_fe and config.model_type in ("wav2vec2", "hubert", "wavlm"):
            # Frontend stats 
            parent = getattr(self.model, config.model_type, None) if config.model_type in ("wav2vec2","hubert","wavlm") else None
            if parent is not None:
                fe = getattr(parent, "feature_extractor", None)
                if fe is not None:
                    fe_total = sum(p.numel() for p in fe.parameters())
                    fe_train = sum(p.numel() for p in fe.parameters() if p.requires_grad)
                    pct = 100.0 * fe_total / total if total else 0.0
                    print(f"\nFeature extractor params: total={fe_total:,} ({pct:.2f}% of model) | trainable={fe_train:,}")
                proj = getattr(parent, "feature_projection", None)
                if proj is not None:
                    proj_total = sum(p.numel() for p in proj.parameters())
                    proj_train = sum(p.numel() for p in proj.parameters() if p.requires_grad)
                    print(f"Feature projection params: total={proj_total:,} | trainable={proj_train:,}\n")
        # List trainable parameter names
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  • {name}: {tuple(param.shape)}")
        # ── Diagnostic: AST position embeddings status ──
        for name, param in self.model.named_parameters():
            if "position_embedding" in name.lower():
                status = "TRAINABLE ⚠️" if param.requires_grad else "FROZEN ✓"
                print(f"\n[DIAG] Position embedding: {name}")
                print(f"       Shape: {tuple(param.shape)} | Status: {status}")
                print(f"       Mean: {param.data.mean().item():.6f} | Std: {param.data.std().item():.6f}")
                norm = param.data.norm().item()
                print(f"       Norm: {norm:.4f} ")
        # Setup metric
        self.train_acc = MulticlassAccuracy(num_classes=len(id2label), average="micro")
        self.val_f1 = MulticlassF1Score(
            num_classes=len(id2label), average="macro"
        )
        self.test_f1 = MulticlassF1Score(num_classes=len(id2label), average="macro")
        self.learning_rate = lr

    def forward(self, **batch):
        return self._forward_impl(**batch)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        preds = output.logits.argmax(dim=-1)
        self.train_acc(preds, batch["labels"])
        self.log(
            "train/loss",
            output.loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["labels"].size(0),
        )
        self.log(
            "train/acc",
            self.train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["labels"].size(0),
        )
        return output.loss

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        train_loss = self.trainer.callback_metrics.get("train/loss", 0.0)
        self._loss_history["train"].append(float(train_loss))
        print(f"Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", end="")

    def validation_step(self, batch, batch_idx):
        """
        Compute validation metrics and log them.
        """
        if getattr(self.trainer, "sanity_checking", False):
            return
        output = self(**batch)
        preds = output.logits.argmax(dim=-1)
        self.val_f1(preds, batch["labels"])
        accuracy = (preds == batch["labels"]).float().mean()
        sync = (self.trainer.num_devices or 1) > 1
        self.log("val/loss", output.loss, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=batch["labels"].size(0))
        self.log("val/acc", accuracy, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=batch["labels"].size(0))
        self.log("val/f1", self.val_f1, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=batch["labels"].size(0))

    def on_validation_epoch_start(self):
        """
        Proactively disable ModelCheckpoint during the first-ever validation run.
        """
        if not self._first_validation_run_completed:
            print("\n[FIX] First validation run detected. Temporarily disabling ModelCheckpoint.")
            for callback in self.trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    callback._original_monitor_key = callback.monitor
                    callback.monitor = "DISABLE_CHECKPOINTING_FOR_THIS_RUN"
                    print(f"[FIX] ModelCheckpoint monitoring '{callback._original_monitor_key}' is now monitoring a dummy key.")

    def on_validation_epoch_end(self):
        """
        Re-enables ModelCheckpoint after the first run.
        """
        if not self._first_validation_run_completed:
            print("[FIX] First validation run finished. Re-enabling ModelCheckpoint for future epochs.")
            for callback in self.trainer.callbacks:
                if isinstance(callback, ModelCheckpoint) and hasattr(callback, '_original_monitor_key'):
                    callback.monitor = callback._original_monitor_key
                    del callback._original_monitor_key
                    print(f"[FIX] ModelCheckpoint monitoring restored to '{callback.monitor}'.")
            
            self._first_validation_run_completed = True
            print(" (ignored pre-training run!!!)")
            return
        
        val_acc = self.trainer.callback_metrics.get("val/acc", 0.0)
        val_loss = self.trainer.callback_metrics.get("val/loss", 0.0)
        self._loss_history["val"].append(float(val_loss))
        
        # Check if best (for print only - callback tracks actual best)
        if float(val_acc) > self.best_val_acc:
            self.best_val_acc = float(val_acc)
            self.best_epoch = int(self.current_epoch)
            best_marker = " *BEST*"
        else:
            best_marker = ""
        
        print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{best_marker}")

    def on_fit_end(self) -> None:
        """Plot train/val loss curves at the end of training."""
        if self.trainer is None or self.trainer.logger is None:
            return
        from pathlib import Path
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        log_dir = Path(self.trainer.logger.log_dir)
        hist = self._loss_history
        if not hist["train"]:
            return
        epochs_train = list(range(len(hist["train"])))
        epochs_val = list(range(len(hist["val"])))
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(epochs_train, hist["train"], "b-", linewidth=2.5, label="Train loss")
        ax.plot(epochs_val, hist["val"], "r-", linewidth=2.5, label="Val loss")
        if self.best_epoch >= 0:
            ax.axvline(x=self.best_epoch, color="purple", linestyle="--",
                       linewidth=2, label=f"Best epoch = {self.best_epoch}")
        ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(loc="best", fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = log_dir / "loss_curves.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved loss curves: {plot_path}")

    def test_step(self, batch, batch_idx):
        """
        Compute test metrics and log them.
        """
        output = self(**batch)
        preds = output.logits.argmax(dim=-1)
        self.test_f1(preds, batch["labels"])
        accuracy = (preds == batch["labels"]).float().mean()
        sync = (self.trainer.num_devices or 1) > 1
        self.log("test/loss", output.loss, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=batch["labels"].size(0))
        self.log("test/acc",  accuracy,  prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=batch["labels"].size(0))
        self.log("test/f1",   self.test_f1, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=batch["labels"].size(0))

    def configure_optimizers(self):
            method = self.hparams.peft_method.lower()
            is_wavlike = self._backbone_model_type in ("wav2vec2", "hubert", "wavlm")
            is_ship = self._dataset in ("deepship", "shipsear")
            is_peft = method not in ("none", "linear")
            if is_wavlike and is_ship and is_peft:
                peft_params = []
                head_params = []
                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    n = name.lower()
                    if "classifier" in n or "modules_to_save" in n:
                        head_params.append(p)
                    else:
                        peft_params.append(p)
                print(f"[LR] Waveform PEFT on ship dataset → PEFT lr={self.learning_rate/5:.1e}, head lr={self.learning_rate:.1e}")
                return torch.optim.AdamW([
                    {"params": peft_params, "lr": self.learning_rate / 5},
                    {"params": head_params, "lr": self.learning_rate},
                ])
            else:
                return torch.optim.AdamW(
                    (p for p in self.parameters() if p.requires_grad),
                    lr=self.learning_rate,
                )
