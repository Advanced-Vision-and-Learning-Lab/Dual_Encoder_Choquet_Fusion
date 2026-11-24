# src/multiencoder_module.py
from typing import Dict, Optional, Tuple, List
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import (
    AutoConfig,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
)
from peft import LoraConfig, IA3Config, get_peft_model, TaskType
from src.adapters import apply_adapter
from src.hpt import apply_hpt  
import json
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from src.model_module import (
    _freeze_classifier_layernorm,
    _unfreeze_last_classifier_linear,
    tie_lora_adapters,
    tie_ia3_vectors,
)

# --------------------------- helpers ---------------------------
def print_trainable_params(module: nn.Module, prefix: str = ""):
    total = 0
    print(f"\n=== Trainable parameters for {prefix or module.__class__.__name__} ===")
    for name, p in module.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            print(f"{prefix}{name:60s} {tuple(p.shape)}  -> {n}")
    print(f"TOTAL trainable params in {prefix or module.__class__.__name__}: {total}\n")

def _freeze_wavlike_feature_extractor(model: nn.Module, model_type: str) -> None:
    if model_type in ("wav2vec2", "hubert", "wavlm"):
        if hasattr(model, "freeze_feature_encoder") and callable(getattr(model, "freeze_feature_encoder")):
            model.freeze_feature_encoder()
            return
        if hasattr(model, "freeze_feature_extractor") and callable(getattr(model, "freeze_feature_extractor")):
            model.freeze_feature_extractor()
            return
        for parent_attr in ("wav2vec2", "hubert", "wavlm"):
            if hasattr(model, parent_attr):
                parent = getattr(model, parent_attr)
                fe = getattr(parent, "feature_extractor", None)
                if fe is not None:
                    for p in fe.parameters():
                        p.requires_grad = False
                break

def _discover_lora_targets(model: nn.Module, target: str):
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
        present = sorted({tok for name, _ in model.named_modules()
                          for tok in TOKENS["query"] if tok in name})
        if not present:
            raise ValueError("LoRA target 'query' matched no modules in this backbone.")
        return present
    raise ValueError(f"Unsupported lora_target='{target}'.")

def _discover_ia3_targets(model: nn.Module, mode: str = "kv_ffn"):
    attn_k, attn_v, ffn = set(), set(), set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        n = name.lower()
        if ("k_proj" in n) or (".key" in n) or (n.endswith(".key")):
            attn_k.add("k_proj" if "k_proj" in n else "key")
        if ("v_proj" in n) or (".value" in n) or (n.endswith(".value")):
            attn_v.add("v_proj" if "v_proj" in n else "value")
        if ("intermediate" in n or "ffn" in n or "output.dense" in n
            or ".mlp." in n or "feedforward" in n):
            ffn.add(name)  
        if "intermediate_dense" in n:
            ffn.add("intermediate_dense")
        if "intermediate.dense" in n:
            ffn.add("intermediate.dense")
    if mode == "v_only":
        target_modules = sorted(attn_v)
        feedforward_modules = []
    elif mode == "kv_ffn":
        target_modules = sorted(attn_k | attn_v | ffn)
        feedforward_modules = sorted(ffn)
    else:
        raise ValueError(f"Unknown ia3_target mode: {mode}")
    return target_modules, feedforward_modules

def _build_base_model(
    model_name: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    freeze_fe: bool,
    ast_max_length: int = 512,
    ast_input_type: str = "logmel",
    mfcc_n_mfcc: int = 32,
    cqt_n_bins: int = 84,
) -> Tuple[nn.Module, "AutoConfig"]:
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id

    if config.model_type == "audio-spectrogram-transformer":
        if ast_input_type == "mfcc":
            config.num_mel_bins = mfcc_n_mfcc
        elif ast_input_type in ["cqt", "vqt"]:
            config.num_mel_bins = cqt_n_bins 
        else:
            config.num_mel_bins = 64
        config.max_length = ast_max_length

    vision_model_types = {"vit"}
    model_type = getattr(config, "model_type", None)
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
    if config.model_type == "audio-spectrogram-transformer":
        import safetensors.torch as st
        from transformers.utils import cached_file
        ckpt_path = cached_file(model_name, "model.safetensors")
        key = "audio_spectrogram_transformer.embeddings.position_embeddings"
        old_pos = st.load_file(ckpt_path)[key]  
        new_pos_param = base_model.audio_spectrogram_transformer.embeddings.position_embeddings
        new_len = new_pos_param.shape[1]
        old_len = old_pos.shape[1]
        if new_len != old_len:
            old_pos_t = old_pos.permute(0, 2, 1)  
            new_pos = F.interpolate(old_pos_t, size=new_len, mode="linear", align_corners=False)
            new_pos = new_pos.permute(0, 2, 1)    
            new_pos_param.data.copy_(new_pos)
            print(f"[INFO] Interpolated AST position embeddings: [{old_len}] → [{new_len}]")

    if config.model_type == "hubert":
        hidden_size = getattr(config, "hidden_size", 768)
        num_labels = config.num_labels
        if hasattr(base_model, "projector"):
            base_model.projector = nn.Identity()
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )
    base_model.gradient_checkpointing_disable()
    if freeze_fe and config.model_type in ("wav2vec2", "hubert", "wavlm"):
        _freeze_wavlike_feature_extractor(base_model, config.model_type)
    allowed_args = set(inspect.signature(base_model.forward).parameters)
    def safe_forward(self_, *args, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in allowed_args}
        if "interpolate_pos_encoding" in allowed_args:
            filtered["interpolate_pos_encoding"] = True
        return base_model.__class__.forward(base_model, *args, **filtered)
    base_model.forward = safe_forward.__get__(base_model, base_model.__class__)
    return base_model, config

def _apply_peft_to_backbone(
    base_model: nn.Module,
    config,
    hparams,
    branch_name: str,
    external_hpt_modules: Optional[List] = None,
) -> Tuple[nn.Module, List]:
    method = hparams.peft_method.lower()
    standard_hpt_modules = []
    model = base_model
    
    if method == "lora":
        target_modules = _discover_lora_targets(model, hparams.lora_target)
        if isinstance(target_modules, list) and len(target_modules) == 0:
            raise ValueError(f"[{branch_name}] No LoRA targets found with lora_target='{hparams.lora_target}'.")
        print(f"[{branch_name}][LoRA] Using target_modules={target_modules}")
        lora_cfg = LoraConfig(
            r=hparams.lora_r,
            lora_alpha=hparams.lora_alpha,
            lora_dropout=hparams.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_cfg)
        if getattr(hparams, "lora_shared", False):
            tie_lora_adapters(model, adapter_name="default")
        for name, p in model.named_parameters():
            if not ("lora_" in name or "modules_to_save" in name or "classifier" in name):
                p.requires_grad = False
        _freeze_classifier_layernorm(model)
        
    elif method == "ia3":
        tmods, fmods = _discover_ia3_targets(model, mode=hparams.ia3_target)
        ia3_cfg = IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=tmods,
            feedforward_modules=fmods,
            inference_mode=False,
        )
        model = get_peft_model(model, ia3_cfg)
        if getattr(hparams, "ia3_shared", False):
            tie_ia3_vectors(model, adapter_name="default", mode=hparams.ia3_target)
        for name, p in model.named_parameters():
            if not ("ia3" in name.lower() or "modules_to_save" in name or "classifier" in name.lower()):
                p.requires_grad = False
        _freeze_classifier_layernorm(model)
        
    elif method == "adapter":
        model = apply_adapter(
            model,
            adapter_dim=hparams.adapter_dim,
            adapter_dropout=hparams.adapter_dropout,
            insertion=hparams.adapter_insertion,
            shared=hparams.adapter_shared,
        )
        _freeze_classifier_layernorm(model)
        _unfreeze_last_classifier_linear(model)
        
    elif method == "hpt":
        model, standard_hpt_modules = apply_hpt(
            model,
            num_bins=hparams.hpt_bins,
            location=hparams.hpt_location,
            shared=hparams.hpt_shared,
            init_rand=getattr(hparams, "init_rand", False),
            freeze_others=True,
            external_hpt_modules=external_hpt_modules,
        )
        _freeze_classifier_layernorm(model)
        _unfreeze_last_classifier_linear(model)
        print(
            f"[{branch_name}] HPT attached: "
            f"location={hparams.hpt_location}, shared={hparams.hpt_shared}, bins={hparams.hpt_bins}"
        )
        
    elif method == "linear":
        for p in model.parameters():
            p.requires_grad = False
        _unfreeze_last_classifier_linear(model)
        _freeze_classifier_layernorm(model)
        
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown peft_method: {method}")
        
    return model, standard_hpt_modules

# --------------------------- Choquet fusion (2 sources) ---------------------------
class Choquet2SourceFusion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        per_class: bool = True,
        k: float = 10.0,
        eps: float = 1e-8,
        init_p: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.per_class = per_class
        self.k = float(k)
        self.eps = float(eps)
        shape = (num_classes,) if per_class else (1,)

        # Clamp init_p to avoid log(0) math domain errors
        p = max(min(init_p, 1.0 - 1e-5), 1e-5)
        logit_1 = math.log(p / (1.0 - p))
        logit_2 = math.log((1.0 - p) / p)  

        #self.raw_g1 = nn.Parameter(torch.zeros(*shape))
        #self.raw_g2 = nn.Parameter(torch.zeros(*shape))
        self.raw_g1 = nn.Parameter(torch.full(shape, float(logit_1)))
        self.raw_g2 = nn.Parameter(torch.full(shape, float(logit_2)))

    def g1(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_g1)

    def g2(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_g2)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        if p1.shape != p2.shape:
            raise ValueError(f"Choquet2SourceFusion expects p1,p2 same shape, got {p1.shape} vs {p2.shape}")
        if p1.dim() != 2:
            raise ValueError(f"Choquet2SourceFusion expects [B,C], got {p1.shape}")
        p1 = p1.clamp(self.eps, 1.0)
        p2 = p2.clamp(self.eps, 1.0)
        g1 = self.g1()
        g2 = self.g2()
        if not self.per_class:
            g1 = g1.expand(p1.size(1))
            g2 = g2.expand(p1.size(1))
        s = torch.sigmoid(self.k * (p1 - p2))  
        c_if_1_ge_2 = p2 + (p1 - p2) * g1  
        c_if_2_gt_1 = p1 + (p2 - p1) * g2  
        fused = s * c_if_1_ge_2 + (1.0 - s) * c_if_2_gt_1
        fused = fused.clamp(self.eps, 1e9)
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return fused

class AverageFusion(nn.Module):
    def __init__(self, num_classes: int, eps: float = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        if p1.shape != p2.shape:
            raise ValueError(f"AverageFusion expects p1,p2 same shape, got {p1.shape} vs {p2.shape}")
        if p1.dim() != 2:
            raise ValueError(f"AverageFusion expects [B,C], got {p1.shape}")
        p1 = p1.clamp(self.eps, 1.0)
        p2 = p2.clamp(self.eps, 1.0)
        fused = (p1 + p2) / 2.0
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return fused

# --------------------------- main module ---------------------------
class MultiEncoderAudioClassifierModule(L.LightningModule):
    def __init__(
        self,
        model_name_wave: str,
        model_name_tf: str,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        peft_method: str = "none",
        freeze_fe: bool = False,
        peft_shared_cross_modal: bool = False,
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
        fusion_type: str = "choque",  
        choque_per_class: bool = True,
        choque_k: float = 10.0,
        choque_init_p: float = 0.5, # if 0.25 means TF-biased (Wave starts at 0.25, TF at 0.75)
        ast_max_length: int = 512,
        dual_branch_loss: bool = False,
        fused_loss_weight: float = 2.0,
        loss_weight_mode: str = "fixed",          
        homoscedastic_init_log_var: float = 0.0,  
        dataset: str = "",
        ast_input_type: str = "logmel",
        mfcc_n_mfcc: int = 32,
        cqt_n_bins: int = 84,
        init_rand: Optional[bool] = None,
    ):
        super().__init__()
        init_rand = hpt_init_rand if init_rand is None else init_rand
        self.save_hyperparameters()
        self.fused_loss_weight = fused_loss_weight
        self._dataset = dataset
        
        wave_base, wave_cfg = _build_base_model(
            model_name_wave, id2label=id2label, label2id=label2id, freeze_fe=freeze_fe, 
            ast_max_length=self.hparams.ast_max_length, ast_input_type=self.hparams.ast_input_type, 
            mfcc_n_mfcc=self.hparams.mfcc_n_mfcc, cqt_n_bins=self.hparams.cqt_n_bins, 
        )
        tf_base, tf_cfg = _build_base_model(
            model_name_tf, id2label=id2label, label2id=label2id, freeze_fe=freeze_fe, 
            ast_max_length=self.hparams.ast_max_length, ast_input_type=self.hparams.ast_input_type, 
            mfcc_n_mfcc=self.hparams.mfcc_n_mfcc, cqt_n_bins=self.hparams.cqt_n_bins, 
        )

        self.wave_config = wave_cfg
        self.tf_config = tf_cfg

        # Apply PEFT to Wave branch and capture standard HPT modules
        self.wave_model, wave_hpt_modules = _apply_peft_to_backbone(
            wave_base, wave_cfg, self.hparams, branch_name="wave"
        )

        # Check cross-modal sharing flag and validate dimensions
        cross_modal_hpt = None
        if getattr(self.hparams, "peft_shared_cross_modal", False) and self.hparams.peft_method in ["hpt", "hptlora"]:
            wave_hidden = getattr(wave_cfg, "hidden_size", None)
            tf_hidden = getattr(tf_cfg, "hidden_size", None)
            if wave_hidden is None or tf_hidden is None or wave_hidden != tf_hidden:
                raise ValueError(
                    f"Cross-modal sharing requires identical hidden sizes. "
                    f"Got Wave: {wave_hidden}, TF: {tf_hidden}"
                )
            print(f"\n[MultiEnc] CROSS-MODAL PEFT SHARING ENABLED! Passing Wave HPT modules directly to TF branch.")
            cross_modal_hpt = wave_hpt_modules

        # Apply PEFT to TF branch, handing off the Wave modules if cross_modal is True
        self.tf_model, _ = _apply_peft_to_backbone(
            tf_base, tf_cfg, self.hparams, branch_name="tf", external_hpt_modules=cross_modal_hpt
        )

        self.fusion_type = fusion_type.lower()
        if self.fusion_type not in ("choque", "avg"):
            raise ValueError(f"Unknown fusion_type={fusion_type}. Valid: 'choque', 'avg'.")
        num_classes = len(id2label)

        self.choque = Choquet2SourceFusion(
            num_classes=num_classes,
            per_class=choque_per_class,
            k=choque_k,
            init_p=choque_init_p)
        
        self.avg_fusion = AverageFusion(num_classes=num_classes)
        
        # ──────────────────────────────────────────────────────────────────────
        # Dual branch loss configuration
        # ──────────────────────────────────────────────────────────────────────
        self.dual_branch_loss = dual_branch_loss
        self.loss_weight_mode = loss_weight_mode
        
        if self.dual_branch_loss:
            self._val_acc_history = []  
            if self.loss_weight_mode == "homoscedastic":
                # Initialize learnable log variances for wave, tf, fused
                self.log_vars = nn.Parameter(torch.full((3,), float(homoscedastic_init_log_var)))
                print(f"\n[MultiEnc] Dual branch loss ENABLED with HOMOSCEDASTIC uncertainty weighting")
            else:
                print(f"\n[MultiEnc] Dual branch loss ENABLED with FIXED weighting (w={fused_loss_weight})")
        
        # --------- Training state tracking ----------
        self.best_epoch = -1
        self.best_val_acc = -1.0
        self._first_validation_run_completed = False
        
        # ── Loss history tracking ──
        self._loss_history = {
            "train": [], "val": [],
            "train_wave": [], "train_tf": [], "train_fused": [],
        }
        self._log_vars_history = []
        
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.learning_rate = lr
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        percent = 100.0 * trainable_params / total_params if total_params else 0.0
        print(f"\n[MultiEnc] Two-encoder model initialized with PEFT method: {peft_method.upper()}")
        print(f"[MultiEnc] Fusion type: {self.fusion_type}")
        print(f"[MultiEnc] Trainable parameters: {trainable_params:,} / {total_params:,} ({percent:.2f}%)")
        self._print_peft_shapes(branch="wave", model=self.wave_model)
        self._print_peft_shapes(branch="tf", model=self.tf_model)
        print_trainable_params(self.wave_model, prefix="wave_model.")
        print_trainable_params(self.tf_model,   prefix="tf_model.")
        print_trainable_params(self.choque,        prefix="choque.")
        print_trainable_params(self.avg_fusion, prefix="avg_fusion.")

    @staticmethod
    def _print_peft_shapes(branch: str, model: nn.Module) -> None:
        lora_shapes = set()
        ia3_shapes = set()
        hpt_shapes = set()
        for name, p in model.named_parameters():
            if "lora_" in name and p.requires_grad:
                lora_shapes.add(tuple(p.shape))
            if "ia3" in name.lower() and p.requires_grad:
                ia3_shapes.add(tuple(p.shape))
            if ("hpt" in name.lower() or "hist" in name.lower() or "gate" in name.lower()) and p.requires_grad:
                hpt_shapes.add(tuple(p.shape))
        if lora_shapes:
            print(f"[MultiEnc][{branch}] LoRA trainable param shapes: {sorted(lora_shapes)}")
        if ia3_shapes:
            print(f"[MultiEnc][{branch}] IA3 trainable param shapes: {sorted(ia3_shapes)}")
        if hpt_shapes:
            print(f"[MultiEnc][{branch}] HPT-related trainable param shapes: {sorted(hpt_shapes)}")
    
    def forward(
        self,
        wave_input_values: Optional[torch.Tensor] = None,
        wave_input_features: Optional[torch.Tensor] = None,
        tf_input_values: Optional[torch.Tensor] = None,
        tf_input_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **unused,
    ):
        wave_kwargs = {}
        if wave_input_values is not None:
            wave_kwargs["input_values"] = wave_input_values
        if wave_input_features is not None:
            wave_kwargs["input_features"] = wave_input_features
            
        tf_kwargs = {}
        if tf_input_values is not None:
            tf_kwargs["input_values"] = tf_input_values
        if tf_input_features is not None:
            tf_kwargs["input_features"] = tf_input_features
            
        wave_out = self.wave_model(
            **wave_kwargs,
            output_hidden_states=True,
            return_dict=True,
        )
        tf_out = self.tf_model(
            **tf_kwargs,
            output_hidden_states=True,
            return_dict=True,
        )
        
        logits_w = wave_out.logits  
        logits_t = tf_out.logits    
        p_w = F.softmax(logits_w, dim=-1)
        p_t = F.softmax(logits_t, dim=-1)
        
        if self.fusion_type == "choque":
            p_fused = self.choque(p_w, p_t)                  
        elif self.fusion_type == "avg":
            p_fused = self.avg_fusion(p_w, p_t)              
        else:
            raise RuntimeError("fusion_type not set correctly")
        
        logp_fused = torch.log(p_fused.clamp_min(1e-8))  
        
        loss = None
        loss_info = {}
        
        if labels is not None:
            if self.dual_branch_loss:
                loss_wave = F.cross_entropy(logits_w, labels)
                loss_tf = F.cross_entropy(logits_t, labels)
                loss_fused = F.nll_loss(logp_fused, labels)

                if self.loss_weight_mode == "homoscedastic":
                    # Dynamic uncertainty weighting calculation
                    precisions = torch.exp(-self.log_vars)
                    
                    loss_w_weighted = loss_wave * precisions[0] + 0.5 * self.log_vars[0]
                    loss_t_weighted = loss_tf * precisions[1] + 0.5 * self.log_vars[1]
                    loss_f_weighted = loss_fused * precisions[2] + 0.5 * self.log_vars[2]
                    
                    loss = loss_w_weighted + loss_t_weighted + loss_f_weighted
                else:
                    # Standard fixed weighting
                    w = self.fused_loss_weight
                    loss = (loss_wave + loss_tf + w * loss_fused) / (2.0 + w)
                
                loss_info = {
                    "loss_wave": loss_wave.detach(),
                    "loss_tf": loss_tf.detach(),
                    "loss_fused": loss_fused.detach(),
                }
            else:
                loss = F.nll_loss(logp_fused, labels)
        
        return {
            "loss": loss,
            "logits": logp_fused,
            "logits_wave": logits_w,
            "logits_tf": logits_t,
            "loss_info": loss_info,
        }
    
    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out["loss"]
        
        preds = out["logits"].argmax(dim=-1)
        self.train_acc(preds, batch["labels"])
        
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
        
        if self.dual_branch_loss and out["loss_info"]:
            info = out["loss_info"]
            self.log("train/loss_wave", info["loss_wave"], on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
            self.log("train/loss_tf", info["loss_tf"], on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
            self.log("train/loss_fused", info["loss_fused"], on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
            
            # Log the current log variances for tensorboard
            if self.loss_weight_mode == "homoscedastic":
                vars_detached = self.log_vars.detach()
                self.log("train/log_var_wave", vars_detached[0], on_step=False, on_epoch=True)
                self.log("train/log_var_tf", vars_detached[1], on_step=False, on_epoch=True)
                self.log("train/log_var_fused", vars_detached[2], on_step=False, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        train_loss = self.trainer.callback_metrics.get("train/loss", 0.0)
        self._loss_history["train"].append(float(train_loss))
        if self.dual_branch_loss:
            lw = self.trainer.callback_metrics.get("train/loss_wave", 0.0)
            lt = self.trainer.callback_metrics.get("train/loss_tf", 0.0)
            lf = self.trainer.callback_metrics.get("train/loss_fused", 0.0)
            self._loss_history["train_wave"].append(float(lw))
            self._loss_history["train_tf"].append(float(lt))
            self._loss_history["train_fused"].append(float(lf))
            
            # Capture log_vars at the end of each epoch for plotting
            if self.loss_weight_mode == "homoscedastic":
                current_log_vars = self.log_vars.detach().cpu().numpy().tolist()
                self._log_vars_history.append({
                    "epoch": int(self.current_epoch),
                    "wave": current_log_vars[0],
                    "tf": current_log_vars[1],
                    "fused": current_log_vars[2]
                })

        print(f"Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", end="")
            
        if self.fusion_type == "choque":
            if self.trainer is None or self.trainer.logger is None:
                print()  
                return
            log_dir = Path(self.trainer.logger.log_dir)
            out_path = log_dir / "fusion_weights.json"
            epoch = int(self.current_epoch)
            g1 = self.choque.g1().detach().cpu()
            g2 = self.choque.g2().detach().cpu()
            payload = {
                "epoch": epoch,
                "type": "choque",
                "g1_measure_singleton_source1(wave_if_p1>=p2)": g1.tolist() if g1.numel() > 1 else float(g1.item()),
                "g2_measure_singleton_source2(tf_if_p2>p1)":     g2.tolist() if g2.numel() > 1 else float(g2.item()),
            }
            if out_path.exists():
                data = json.loads(out_path.read_text())
            else:
                data = []
            data.append(payload)
            out_path.write_text(json.dumps(data, indent=2))
        
        print()  
    
    def on_validation_epoch_start(self):
        if not self._first_validation_run_completed:
            print("\n[FIX] First validation run detected. Temporarily disabling ModelCheckpoint.")
            for callback in self.trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    callback._original_monitor_key = callback.monitor
                    callback.monitor = "DISABLE_CHECKPOINTING_FOR_THIS_RUN"
                    print(f"[FIX] ModelCheckpoint monitoring '{callback._original_monitor_key}' is now monitoring a dummy key.")
    
    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        logits = out["logits"]
        labels = batch["labels"]
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.val_f1(preds, labels)
        sync = (self.trainer.num_devices or 1) > 1
        if out["loss"] is not None:
            self.log("val/loss", out["loss"], prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
        self.log("val/acc", acc, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
        self.log("val/f1", self.val_f1, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
        
        if self.dual_branch_loss:
            preds_wave = out["logits_wave"].argmax(dim=-1)
            preds_tf = out["logits_tf"].argmax(dim=-1)
            acc_wave = (preds_wave == labels).float().mean()
            acc_tf = (preds_tf == labels).float().mean()
            self.log("val/acc_wave", acc_wave, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
            self.log("val/acc_tf", acc_tf, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
    
    def on_validation_epoch_end(self):
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
        
        if float(val_acc) > self.best_val_acc:
            self.best_val_acc = float(val_acc)
            self.best_epoch = int(self.current_epoch)
            best_marker = " *BEST*"
        else:
            best_marker = ""
        
        extra_info = ""
        if self.dual_branch_loss:
            val_acc_wave = self.trainer.callback_metrics.get("val/acc_wave", 0.0)
            val_acc_tf = self.trainer.callback_metrics.get("val/acc_tf", 0.0)
            extra_info = f" | Wave Acc: {val_acc_wave:.4f} | TF Acc: {val_acc_tf:.4f}"
        
            self._val_acc_history.append({
                "epoch": int(self.current_epoch),
                "acc_fused": float(val_acc),
                "acc_wave": float(val_acc_wave),
                "acc_tf": float(val_acc_tf),
            })
        print(f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{extra_info}{best_marker}")
    
    def test_step(self, batch, batch_idx):
        out = self(**batch)
        logits = out["logits"]
        labels = batch["labels"]
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.test_f1(preds, labels)
        sync = (self.trainer.num_devices or 1) > 1
        if out["loss"] is not None:
            self.log("test/loss", out["loss"], prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
        self.log("test/acc", acc, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
        self.log("test/f1", self.test_f1, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
        
        if self.dual_branch_loss:
            preds_wave = out["logits_wave"].argmax(dim=-1)
            preds_tf = out["logits_tf"].argmax(dim=-1)
            acc_wave = (preds_wave == labels).float().mean()
            acc_tf = (preds_tf == labels).float().mean()
            self.log("test/acc_wave", acc_wave, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
            self.log("test/acc_tf", acc_tf, prog_bar=False, on_epoch=True, sync_dist=sync, batch_size=labels.size(0))
    
    def on_fit_end(self) -> None:
        if self.trainer is None or self.trainer.logger is None:
            return
        
        log_dir = Path(self.trainer.logger.log_dir)
        
        self._plot_loss_curves(log_dir)
        if self.dual_branch_loss and hasattr(self, '_val_acc_history') and self._val_acc_history:
            self._plot_val_accuracies(log_dir)
        if self.fusion_type == "choque":
            self._plot_choquet_weights(log_dir)
            
        # Plot homoscedastic weights if enabled
        if self.dual_branch_loss and getattr(self, "loss_weight_mode", "fixed") == "homoscedastic":
            self._plot_homoscedastic_weights(log_dir)

    def _plot_loss_curves(self, log_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hist = self._loss_history
        if not hist["train"]:
            return
        epochs_t = list(range(len(hist["train"])))
        epochs_v = list(range(len(hist["val"])))
        fig, ax = plt.subplots(figsize=(5, 3))

        if self.dual_branch_loss and hist["train_wave"]:
            # If homoscedastic, weights are applied dynamically during computation
            if self.loss_weight_mode == "homoscedastic":
                ax.plot(epochs_t, hist["train_wave"], "-", color="#1f77b4", linewidth=2.0, label="Wave loss (raw)")
                ax.plot(epochs_t, hist["train_tf"], "-", color="#ff7f0e", linewidth=2.0, label="TF loss (raw)")
                ax.plot(epochs_t, hist["train_fused"], "-", color="#2ca02c", linewidth=2.0, label="Fused loss (raw)")
                ax.plot(epochs_t, hist["train"], "k-", linewidth=2.5, label="Total train loss (weighted)")
            else:
                w = self.hparams.fused_loss_weight
                denom = 2.0 + w
                w_wave  = [v * (1.0 / denom) for v in hist["train_wave"]]
                w_tf    = [v * (1.0 / denom) for v in hist["train_tf"]]
                w_fused = [v * (w / denom) for v in hist["train_fused"]]
                
                ax.plot(epochs_t, w_wave,  "-",  color="#1f77b4", linewidth=2.0, label=f"Wave loss (x {1.0/denom:.2f})")
                ax.plot(epochs_t, w_tf,    "-",  color="#ff7f0e", linewidth=2.0, label=f"TF loss (x {1.0/denom:.2f})")
                ax.plot(epochs_t, w_fused, "-",  color="#2ca02c", linewidth=2.0, label=f"Fused loss (x {w/denom:.2f})")
                ax.plot(epochs_t, hist["train"], "k-", linewidth=2.5, label="Total train loss")

        else:
            ax.plot(epochs_t, hist["train"], "b-", linewidth=2.5, label="Train loss")
            
        ax.plot(epochs_v, hist["val"], "r-", linewidth=2.5, label="Val loss")
        if self.best_epoch >= 0:
            ax.axvline(x=self.best_epoch, color="purple", linestyle="--", linewidth=2, label=f"Best epoch = {self.best_epoch}")
        ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(loc="best", fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = log_dir / "loss_curves.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[MultiEnc] Saved loss curves: {plot_path}")

    def _plot_val_accuracies(self, log_dir: Path) -> None:
        import matplotlib.pyplot as plt
        data = self._val_acc_history
        if not data:
            return
        
        epochs = [d["epoch"] for d in data]
        acc_wave = [d["acc_wave"] for d in data]
        acc_tf = [d["acc_tf"] for d in data]
        acc_fused = [d["acc_fused"] for d in data]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, acc_wave, 'b-', linewidth=3, label='Wave branch')
        ax.plot(epochs, acc_tf, 'r-', linewidth=3, label='TF branch')
        ax.plot(epochs, acc_fused, 'g-', linewidth=3, label='Fused')
        
        ax.axvline(x=self.best_epoch, color='purple', linestyle='--', linewidth=2.5, label=f'Best epoch = {self.best_epoch}')
        
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Validation Accuracy", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(epochs), max(epochs))
        ax.set_ylim(0, 1)
        
        fig.tight_layout()
        plot_path = log_dir / "val_accuracy_branches_plot.png"
        fig.savefig(plot_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"[MultiEnc] Saved validation accuracy plot: {plot_path}")

    def _plot_choquet_weights(self, log_dir: Path) -> None:
        import matplotlib.pyplot as plt
        json_path = log_dir / "fusion_weights.json"
        if not json_path.exists():
            return
        
        data = json.loads(json_path.read_text())
        if not data:
            return
        
        epochs = [d["epoch"] for d in data]
        g1_key = "g1_measure_singleton_source1(wave_if_p1>=p2)"
        g2_key = "g2_measure_singleton_source2(tf_if_p2>p1)"
        
        g1_all = [d[g1_key] for d in data]
        g2_all = [d[g2_key] for d in data]
        
        is_per_class = isinstance(g1_all[0], list)
        
        if is_per_class:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            num_classes = len(g1_all[0])
            
            for c in range(num_classes):
                g1_c = [g[c] for g in g1_all]
                g2_c = [g[c] for g in g2_all]
                label = self.hparams.id2label.get(c, f"Class {c}") if hasattr(self.hparams, 'id2label') else f"Class {c}"
                axes[0].plot(epochs, g1_c, label=label, alpha=0.8, linewidth=1.5)
                axes[1].plot(epochs, g2_c, label=label, alpha=0.8, linewidth=1.5)
            
            for ax in axes:
                ax.axvline(x=self.best_epoch, color='purple', linestyle='--', linewidth=2.5, label=f'Best epoch = {self.best_epoch}')
                ax.set_xlabel("Epoch", fontsize=14)
                ax.set_ylabel("Weight", fontsize=14)
                ax.tick_params(axis='both', labelsize=12)
                ax.set_ylim(0, 1)
                ax.set_xlim(min(epochs), max(epochs))
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=10)
            
            axes[0].set_ylabel("g1 (wave preference)", fontsize=14)
            axes[1].set_ylabel("g2 (TF preference)", fontsize=14)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, g1_all, 'b-', linewidth=3, label='g1 (wave preference)')
            ax.plot(epochs, g2_all, 'r-', linewidth=3, label='g2 (TF preference)')
            ax.axvline(x=self.best_epoch, color='purple', linestyle='--', linewidth=2.5)
            
            ax.set_xlabel("Epoch", fontsize=14, fontweight='bold')
            ax.set_ylabel("Fuzzy Measure Weight", fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            ax.set_xlim(min(epochs), max(epochs))
        
        fig.tight_layout()
        plot_path = log_dir / "fusion_weights_plot.png"
        fig.savefig(plot_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"[MultiEnc] Saved fusion weights plot: {plot_path}")

    def _plot_homoscedastic_weights(self, log_dir: Path) -> None:
        """Generate plots showing homoscedastic log-variance and precision weights over time."""
        if not hasattr(self, "_log_vars_history") or not self._log_vars_history:
            return
            
        import matplotlib.pyplot as plt
        import json
        import math
        
        epochs = [d["epoch"] for d in self._log_vars_history]
        wave_vars = [d["wave"] for d in self._log_vars_history]
        tf_vars = [d["tf"] for d in self._log_vars_history]
        fused_vars = [d["fused"] for d in self._log_vars_history]
        
        # Calculate the actual weights: exp(-s)
        wave_weights = [math.exp(-v) for v in wave_vars]
        tf_weights = [math.exp(-v) for v in tf_vars]
        fused_weights = [math.exp(-v) for v in fused_vars]

        # --- Plot 1: Log-Variance (The original plot) ---
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(epochs, wave_vars, 'b-', linewidth=2.5, label='Wave Log-Var')
        ax1.plot(epochs, tf_vars, 'r-', linewidth=2.5, label='TF Log-Var')
        ax1.plot(epochs, fused_vars, 'g-', linewidth=2.5, label='Fused Log-Var')
        
        if self.best_epoch >= 0:
            ax1.axvline(x=self.best_epoch, color='purple', linestyle='--', linewidth=2.5, label=f'Best epoch = {self.best_epoch}')
            
        ax1.set_xlabel("Epoch", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Log-Variance ($s = \log \sigma^2$)", fontsize=14, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=12)
        ax1.legend(loc='best', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        fig1.tight_layout()
        plot_path1 = log_dir / "homoscedastic_log_vars.png"
        fig1.savefig(plot_path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # --- Plot 2: Actual Loss Weights ---
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(epochs, wave_weights, 'b-', linewidth=2.5, label='Wave Weight')
        ax2.plot(epochs, tf_weights, 'r-', linewidth=2.5, label='TF Weight')
        ax2.plot(epochs, fused_weights, 'g-', linewidth=2.5, label='Fused Weight')
        
        if self.best_epoch >= 0:
            ax2.axvline(x=self.best_epoch, color='purple', linestyle='--', linewidth=2.5, label=f'Best epoch = {self.best_epoch}')
            
        ax2.set_xlabel("Epoch", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Loss Multiplier Weight ($\exp(-s)$)", fontsize=14, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=12)
        ax2.legend(loc='best', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        fig2.tight_layout()
        plot_path2 = log_dir / "homoscedastic_exp_weights.png"
        fig2.savefig(plot_path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Save JSON history
        json_path = log_dir / "homoscedastic_weights.json"
        json_path.write_text(json.dumps(self._log_vars_history, indent=2))
        print(f"[MultiEnc] Saved homoscedastic plots: {plot_path1.name} and {plot_path2.name}")


    def configure_optimizers(self):
        method = self.hparams.peft_method.lower()
        is_peft = method not in ("none", "linear")
        is_ship = self._dataset in ("deepship", "shipsear")
        wave_is_wavlike = self.wave_config.model_type in ("wav2vec2", "hubert", "wavlm")
        tf_is_wavlike = self.tf_config.model_type in ("wav2vec2", "hubert", "wavlm")
        wave_lr = (self.learning_rate / 5) if (wave_is_wavlike and is_ship and is_peft) else self.learning_rate
        tf_lr = (self.learning_rate / 5) if (tf_is_wavlike and is_ship and is_peft) else self.learning_rate
        
        param_groups = []
        seen_params = set()  # Track memory IDs to prevent double-adding shared weights
        
        for name, p in self.wave_model.named_parameters():
            if not p.requires_grad or id(p) in seen_params:
                continue
            n = name.lower()
            lr = self.learning_rate if ("classifier" in n or "modules_to_save" in n) else wave_lr
            param_groups.append({"params": [p], "lr": lr})
            seen_params.add(id(p))
            
        for name, p in self.tf_model.named_parameters():
            if not p.requires_grad or id(p) in seen_params:
                continue
            n = name.lower()
            lr = self.learning_rate if ("classifier" in n or "modules_to_save" in n) else tf_lr
            param_groups.append({"params": [p], "lr": lr})
            seen_params.add(id(p))
            
        for p in self.choque.parameters():
            if p.requires_grad and id(p) not in seen_params:
                param_groups.append({"params": [p], "lr": self.learning_rate})
                seen_params.add(id(p))
                
        for p in self.avg_fusion.parameters():
            if p.requires_grad and id(p) not in seen_params:
                param_groups.append({"params": [p], "lr": self.learning_rate})
                seen_params.add(id(p))
                
        # ── Add learnable log-variances to optimizer if homoscedastic is active ──
        if getattr(self, "dual_branch_loss", False) and getattr(self, "loss_weight_mode", "fixed") == "homoscedastic":
            if id(self.log_vars) not in seen_params:
                param_groups.append({"params": [self.log_vars], "lr": self.learning_rate})
                seen_params.add(id(self.log_vars))
                print(f"[LR] Added homoscedastic log_vars to optimizer with lr={self.learning_rate:.1e}")

        print(f"[LR] Wave PEFT lr={wave_lr:.1e}, TF PEFT lr={tf_lr:.1e}, head/fusion lr={self.learning_rate:.1e}")
        return torch.optim.AdamW(param_groups)
