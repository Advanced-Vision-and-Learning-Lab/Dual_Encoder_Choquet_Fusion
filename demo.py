# demo.py
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
torch.set_float32_matmul_precision("medium")
# Hugging Face cache to a local folder
CACHE_DIR = Path(__file__).parent / "cache_for_HF"
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
import lightning as L
import inspect
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_warning()
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.INFO)
from datasets.DeepShipDataModule import DSAudioDataModule
from datasets.ShipsEarDataModule import ShipsEarDataModule
from src.model_module import AudioClassifierModule
from utils.metrics import aggregate_and_save
from src.multiencoder_module import MultiEncoderAudioClassifierModule
from utils.profiling import profile_inference, profile_train_step, profile_flops_bsz1
from utils.run_evaluation_plots import (
    collect_test_predictions, save_per_run_confusion_matrix,
    save_aggregated_confusion_matrix, compute_roc_data, save_per_run_roc,
    save_aggregated_roc, save_aggregated_roc_per_class,
    save_aggregated_fusion_weights,
)

def _build_run_name(args) -> str:
    parts = [args.peft_method]
    if args.peft_method == "lora":
        tag = f"r{args.lora_r}-t{args.lora_target}"
        if getattr(args, "lora_shared", False):
            tag += "-sh"
        parts.append(tag)
    elif args.peft_method == "ia3":
        tag = args.ia3_target
        if getattr(args, "ia3_shared", False):
            tag += "-sh"
        parts.append(tag)
    elif args.peft_method == "adapter":
        tag = f"d{args.adapter_dim}-{args.adapter_insertion}"
        if args.adapter_shared:
            tag += "-sh"
        parts.append(tag)
    elif args.peft_method == "hpt":
        tag = f"b{args.hpt_bins}-{args.hpt_location}"
        if args.hpt_shared:
            tag += "-sh"
        if getattr(args, "peft_shared_cross_modal", False):
            tag += "-crossmodal"
        if getattr(args, "init_rand", False):
            tag += "-init_rand"
        parts.append(tag)
    elif args.peft_method == "linear":
        parts.append("headonly")
    elif args.peft_method == "none":
        parts.append("full")
    
    # Few-shot suffix
    k = getattr(args, "k_shot_per_class", 0)
    if k and k > 0:
        if len(parts) >= 2:
            parts[-1] = parts[-1] + f"_fs{k}"
        else:
            parts.append(f"fs{k}")
    
    enc_mode = getattr(args, "encoder_mode", "single")
    if enc_mode == "dual":
        # Append the weight to dualenc if dual_branch_loss is active
        if getattr(args, "dual_branch_loss", False):
            lw_mode = getattr(args, "loss_weight_mode", "fixed")
            if lw_mode == "homoscedastic":
                parts.append("dualenc_homo")
            else:
                w = getattr(args, "fused_loss_weight", 2.0)
                parts.append(f"dualenc_w{w:g}")
        else:
            parts.append("dualenc")
            
        fusion = getattr(args, "fusion_type", "choque")
        if fusion != "choque":  
            parts.append(f"fusion_{fusion}")
        if getattr(args, "dual_branch_loss", False):
            parts.append("dbl")

    if getattr(args, "dataset", None) == "deepship":
        src = getattr(args, "deepship_split_source", "default")
        src_short = "std" if src == "std_deepship_data_split" else ("rnd" if src == "random" else "def")
        parts.append(f"src{src_short}")   # -> "srcstd" or "srcdef"
    # ShipsEar split strategy tag
    if getattr(args, "dataset", None) == "shipsear":
        ss = getattr(args, "shipsear_split_strategy", "temporal")
        if ss == "random":
            parts.append("rsplit")   # or "randomsplit"

    # AST input representation tag (logmel vs mfcc) — affects only ship datasets / AST branches
    if getattr(args, "dataset", None) in ("deepship", "shipsear"):
        parts.append(f"{getattr(args, 'ast_input_type', 'logmel')}")
            
    return "__".join(parts)

def _group_dir(args) -> Path:
    if getattr(args, "encoder_mode", "single") == "dual":
        wave = getattr(args, "model_name_wave", None) or args.model_name
        tf   = getattr(args, "model_name_tf", None) or args.model_name
        model_safe = f"{wave.replace('/', '-')}__{tf.replace('/', '-')}"
    else:
        model_safe = args.model_name.replace("/", "-")
    return Path("lightning_logs") / f"{args.dataset}__{model_safe}"

def _ensure_group_dir(args):
    d = _group_dir(args)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _aggregate_metrics_dicts(dicts):
    from collections import defaultdict
    buckets = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)) and v is not None:
                buckets[k].append(float(v))
    out = {}
    for k, vals in buckets.items():
        if len(vals) == 0:
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "n":    int(len(vals)),
        }
    return out

def run_single(args, dm, label2id, id2label):
    assert args.gpus in (0, 1), "Single-GPU or CPU expected."
    exp_name  = _build_run_name(args)
    group_dir = _ensure_group_dir(args)
    
    # Setup logger
    logger = TensorBoardLogger(
            save_dir=str(group_dir),
            name=exp_name,
            version=f"run{args.run_idx}")
    
    # Setup callbacks
    ckpt_cb = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="best-model"
    )
    stop_cb = EarlyStopping(monitor="val/loss", patience=30, mode="min")
    
    callbacks_list = [ckpt_cb, stop_cb]

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        precision="16-mixed" if args.gpus > 0 else 32,
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=False,
        callbacks=callbacks_list,
        num_sanity_val_steps=0,
    )

    # Determine which Model Class to use based on encoder_mode
    encoder_mode = getattr(args, "encoder_mode", "single")
    if encoder_mode == "dual":
        ModelClass = MultiEncoderAudioClassifierModule
    else:
        ModelClass = AudioClassifierModule

    # Build the initial model instance for training
    model_args = vars(args).copy()
    model_args.update({
        "id2label": id2label,
        "label2id": label2id
    })

    # Filter only the arguments that are valid for the selected ModelClass's __init__
    sig = inspect.signature(ModelClass.__init__)
    valid_args = {k: v for k, v in model_args.items() if k in sig.parameters}
    model = ModelClass(**valid_args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
    model.to(device)

    # Train the model
    trainer.fit(model=model, datamodule=dm)

    # Get the path to the best checkpoint.
    best_ckpt_path = ckpt_cb.best_model_path
    print(f"\nTraining finished. Best checkpoint path: {best_ckpt_path}")
    if not best_ckpt_path:
        print("\n[WARNING] No best checkpoint found. Skipping test phase.")
        return None, {}

    # Load the best model from the checkpoint file.
    print(f"Loading best model from checkpoint for testing and reporting...")
    encoder_mode = getattr(args, "encoder_mode", "single")
    if encoder_mode == "dual": ModelClass = MultiEncoderAudioClassifierModule
    else: ModelClass = AudioClassifierModule
    
    best_model = ModelClass.load_from_checkpoint(best_ckpt_path)

    profiling_metrics = {}
    try:
        # Get a sample batch for profiling
        dm.setup(stage="fit")
        sample_batch = next(iter(dm.val_dataloader()))
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
        best_model.to(device)
        best_model.eval()
        
        # FLOPs (batch size 1)
        try:
            flops_result = profile_flops_bsz1(best_model, sample_batch)
            profiling_metrics.update(flops_result)
            print(f"  FLOPs (bsz=1): {flops_result.get('gflops_bsz1', 'N/A')} GFLOPs")
        except Exception as e:
            print(f"[WARNING] FLOPs profiling failed: {e}")
            profiling_metrics["gflops_bsz1"] = None
        
        # Inference latency
        try:
            infer_result = profile_inference(
                model=best_model,
                batch=sample_batch,
                device=device,
                repeats=50,
                warmup=10,
                use_mixed_precision=(args.gpus > 0),
            )
            profiling_metrics["latency_ms_per_sample"] = infer_result.get("infer_ms_per_sample_mean")
            profiling_metrics["infer_throughput_sps_mean"] = infer_result.get("infer_throughput_sps_mean")
            profiling_metrics["infer_peak_alloc_mb"] = infer_result.get("infer_peak_alloc_mb")
            print(f"  Inference latency: {infer_result.get('infer_ms_per_sample_mean', 'N/A'):.3f} ms/sample")
        except Exception as e:
            print(f"[WARNING] Inference profiling failed: {e}")
            profiling_metrics["latency_ms_per_sample"] = None
            profiling_metrics["infer_throughput_sps_mean"] = None
        
        # Training step profiling
        try:
            train_result = profile_train_step(
                model=best_model,
                batch=sample_batch,
                device=device,
                lr=args.lr,
                warmup_steps=3,
                repeats=3,
                use_mixed_precision=(args.gpus > 0),
            )
            profiling_metrics["train_ms_per_sample"] = train_result.get("train_ms_per_sample_mean")
            profiling_metrics["train_throughput_sps_mean"] = train_result.get("train_throughput_sps_mean")
            profiling_metrics["train_peak_alloc_mb"] = train_result.get("train_peak_alloc_mb")
            print(f"  Train throughput: {train_result.get('train_throughput_sps_mean', 'N/A'):.1f} samples/s")
        except Exception as e:
            print(f"[WARNING] Train step profiling failed: {e}")
            profiling_metrics["train_throughput_sps_mean"] = None
            profiling_metrics["train_peak_alloc_mb"] = None
        
    except Exception as e:
        print(f"[WARNING] Profiling setup failed: {e}")

    test_trainer = L.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        precision="16-mixed" if args.gpus > 0 else 32,
        logger=False,
        enable_progress_bar=False
    )
    
    # Run the final test and get the test accuracy.
    print("Running final test...")
    test_results = test_trainer.test(model=best_model, datamodule=dm, verbose=False)[0]
    test_acc = float(test_results.get("test/acc", 0.0))
    test_f1 = float(test_results.get("test/f1", 0.0))

    # ── Per-run confusion matrix & ROC ──
    run_eval_data = {}
    try:
        dm.setup(stage="test")
        test_dl = dm.test_dataloader()
        encoder_mode = getattr(args, "encoder_mode", "single")
        preds, labels_arr, logits_arr = collect_test_predictions(
            best_model, test_dl, device, encoder_mode=encoder_mode,
        )
        # Class names
        class_names = [id2label[i] for i in range(len(id2label))]
        num_classes = len(class_names)
        # Determine per-run save directory (same as logger log_dir)
        if logger.log_dir:
            run_plot_dir = Path(logger.log_dir)
        else:
            run_plot_dir = group_dir / exp_name / f"run{args.run_idx}"
        run_plot_dir.mkdir(parents=True, exist_ok=True)
        # Per-run confusion matrix
        cm_pct = save_per_run_confusion_matrix(
            preds, labels_arr, class_names,
            outfile=run_plot_dir / "confusion_matrix.png",
        )
        # Per-run ROC
        roc_data = compute_roc_data(labels_arr, logits_arr, num_classes)
        save_per_run_roc(roc_data, class_names, outfile=run_plot_dir / "roc_curve.png")
        macro_auc = float(roc_data.get("macro_auc", 0.0))
        run_eval_data = {
            "cm_percent": cm_pct,
            "roc_data": roc_data,
            "run_log_dir": str(run_plot_dir),
        }
    except Exception as e:
        print(f"[WARNING] Per-run evaluation plots failed: {e}")

    best_epoch = best_model.best_epoch
    best_val_acc = best_model.best_val_acc
    total_params = sum(p.numel() for p in best_model.parameters())
    trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    
    # Extract fusion weights from best model (dual encoder only)
    fusion_weights_best = {}
    if getattr(args, "encoder_mode", "single") == "dual":
        if hasattr(best_model, "fusion_type"):
            if best_model.fusion_type == "choque":
                fusion_weights_best = {
                    "type": "choque",
                    "g1_per_class": best_model.choque.g1().detach().cpu().tolist(),
                    "g2_per_class": best_model.choque.g2().detach().cpu().tolist(),
                }
            elif best_model.fusion_type == "prob_weighted":
                alpha = best_model.prob_weighted.alpha().detach().cpu()
                fusion_weights_best = {
                    "type": "prob_weighted",
                    "alpha": alpha.tolist() if alpha.numel() > 1 else float(alpha.item()),
                }

    best_epoch = getattr(model, 'best_epoch', -1)
    best_val_acc = getattr(model, 'best_val_acc', -1.0)
    
    metrics = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_percent": float(100.0 * trainable_params / total_params),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "best_val_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "val_accuracy": float(best_val_acc),
        "test_macro_auc": float(macro_auc) if "macro_auc" in locals() else None,
        **profiling_metrics,
    }
    if fusion_weights_best:
        metrics["fusion_weights_best_epoch"] = fusion_weights_best
    metrics["_run_eval_data"] = run_eval_data
    return best_ckpt_path, metrics

def build_dm_and_labels(args):
    if args.dataset == "shipsear":
        encoder_mode = getattr(args, "encoder_mode", "single")
        if encoder_mode == "single":
            dm = ShipsEarDataModule(
                shipsear_root=Path(args.shipsear_root),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sample_rate=args.shipsear_sr,
                segment_length_s=args.shipsear_seglen,
                model_name=args.model_name,
                encoder_mode="single",
                augment=not args.no_augment,
                split_strategy=args.shipsear_split_strategy,
                ast_input_type=args.ast_input_type,
                mfcc_n_mfcc=args.mfcc_n_mfcc,
                mfcc_n_mels=args.mfcc_n_mels,
                cqt_n_bins=args.cqt_n_bins,
                norm_type=args.norm_type,
            )
        elif encoder_mode == "dual":
            wave_name = getattr(args, "model_name_wave", None) or args.model_name
            tf_name   = getattr(args, "model_name_tf", None) or args.model_name
            dm = ShipsEarDataModule(
                shipsear_root=Path(args.shipsear_root),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sample_rate=args.shipsear_sr,
                segment_length_s=args.shipsear_seglen,
                model_name=tf_name,
                encoder_mode="dual",
                augment=not args.no_augment,
                model_name_wave=wave_name,
                model_name_tf=tf_name,
                split_strategy=args.shipsear_split_strategy,
                ast_input_type=args.ast_input_type,
                mfcc_n_mfcc=args.mfcc_n_mfcc,
                mfcc_n_mels=args.mfcc_n_mels,
                cqt_n_bins=args.cqt_n_bins,
                norm_type=args.norm_type,
            )
        dm.prepare_data()
        dm.setup(stage="fit")
        if hasattr(dm, "actual_time_frames"):
            args.ast_max_length = dm.actual_time_frames
        label2id = dm.label2id
        id2label = dm.id2label
        return dm, label2id, id2label
    
    encoder_mode = getattr(args, "encoder_mode", "single")
    if encoder_mode == "single":
        dm = DSAudioDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=args.deepship_sr,
            segment_length_s=args.deepship_seglen,
            split_level=args.deepship_split,
            split_source=args.deepship_split_source,
            data_root=Path(args.deepship_root),
            model_name=args.model_name,
            k_shot_per_class=getattr(args, "k_shot_per_class", 0),
            encoder_mode="single",
            augment=not args.no_augment, 
            ast_input_type=args.ast_input_type,
            mfcc_n_mfcc=args.mfcc_n_mfcc,
            mfcc_n_mels=args.mfcc_n_mels,
            cqt_n_bins=args.cqt_n_bins,
            norm_type=args.norm_type,
        )
    elif encoder_mode == "dual":
        wave_name = getattr(args, "model_name_wave", None) or args.model_name
        tf_name   = getattr(args, "model_name_tf", None) or args.model_name
        dm = DSAudioDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=args.deepship_sr,
            segment_length_s=args.deepship_seglen,
            split_level=args.deepship_split,
            split_source=args.deepship_split_source,
            data_root=Path(args.deepship_root),
            model_name=tf_name,
            k_shot_per_class=getattr(args, "k_shot_per_class", 0),
            encoder_mode="dual",
            augment=not args.no_augment, 
            model_name_wave=wave_name,
            model_name_tf=tf_name,
            ast_input_type=args.ast_input_type,
            mfcc_n_mfcc=args.mfcc_n_mfcc,
            mfcc_n_mels=args.mfcc_n_mels,
            cqt_n_bins=args.cqt_n_bins,
            norm_type=args.norm_type,
        )
    else:
        raise ValueError(f"Unknown encoder_mode: {encoder_mode}")
    
    dm.prepare_data()
    dm.setup(stage="fit")
    if hasattr(dm, "actual_time_frames"):
        args.ast_max_length = dm.actual_time_frames
    if hasattr(dm, "print_dataset_stats"):
        dm.print_dataset_stats()
    label2id = dm.label2id
    id2label = dm.id2label
    return dm, label2id, id2label

def main():
    parser = argparse.ArgumentParser("Classification with PEFT")
    parser.add_argument("--dataset",choices=["deepship", "shipsear"], default="shipsear")
    parser.add_argument("--encoder_mode", choices=["single", "dual"], default="single",
                        help="single: one encoder; dual: wave + TF.")
    parser.add_argument("--model_name_wave", type=str, default=None,
                        help="Backbone for waveform encoder (used when encoder_mode=dual).")
    parser.add_argument("--model_name_tf", type=str, default=None,
                        help="Backbone for TF encoder (used when encoder_mode=dual).")
    
    parser.add_argument("--fusion_type", choices=["choque", "avg"], default="choque",help="Fusion strategy for dual encoder.")
    
    parser.add_argument("--dual_branch_loss", action="store_true", help="Enable per-branch losses.")
    
    parser.add_argument("--fused_loss_weight", type=float, default=2.0, help="Weight for the fused loss component.")

    parser.add_argument("--norm_type", type=str, default="ast", choices=["ast", "sample"])
    
    parser.add_argument("--ast_input_type", choices=["logmel", "mfcc", "cqt", "vqt"], default="logmel")
    
    parser.add_argument("--cqt_n_bins", type=int, default=84,help="Number of frequency bins for CQT or VQT.")
    parser.add_argument("--mfcc_n_mfcc", type=int, default=32, help="Number of MFCC coefficients (feature dim fed to AST).")
    parser.add_argument("--mfcc_n_mels", type=int, default=128, help="Number of mel bins for MFCC computation.")
    
    parser.add_argument("--loss_weight_mode", choices=["fixed", "homoscedastic"], default="fixed",
                        help="Loss weighting: 'fixed' uses preset weights, 'homoscedastic' uses uncertainty-based weighting (Kendall et al., 2018).")
    parser.add_argument("--fixed_branch_weight", type=float, default=0.25,
                        help="Weight for each branch loss in fixed mode (default: 0.25).")
    parser.add_argument("--fixed_mixed_weight", type=float, default=0.25,
                        help="Weight for mixed classifier loss in fixed mode (default: 0.25).")
    parser.add_argument("--homoscedastic_init_log_var", type=float, default=0.0,
                        help="Initial log-variance for homoscedastic weighting. 0.0 means σ²=1 (equal initial weighting).")
    
    parser.add_argument("--fusion_mode", choices=["average", "gated"], default="average",
                        help="Fusion strategy: 'average' for simple mean, 'gated' for learnable scalar weights.")
    
    parser.add_argument("--no_augment", action="store_true", help="Disable audio augmentation for ship datasets (enabled by default)")
                        
    # Model/PEFT
    parser.add_argument("--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--peft_method", choices=["none", "linear", "lora", "adapter", "ia3", "hpt"], default="hpt")
    parser.add_argument("--freeze_fe", action="store_true", help="Freeze raw feature extractor for wav2vec2/hubert")
    parser.add_argument("--lora_target", choices=["all", "query", "qv"], default="query")
    parser.add_argument("--ia3_target", choices=["kv_ffn", "v_only"], default="kv_ffn")
    parser.add_argument("--ia3_shared", action="store_true")
    parser.add_argument("--adapter_dim", type=int, default=16)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_insertion", choices=["ffn", "mhsa"], default="mhsa")
    parser.add_argument("--adapter_shared", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_shared", action="store_true")
    parser.add_argument("--hpt_bins", type=int, default=4)
    parser.add_argument("--hpt_location", choices=["mhsa", "ffn", "mhsa_ffn"], default="mhsa")
    parser.add_argument("--hpt_shared", action="store_true")
    parser.add_argument("--peft_shared_cross_modal", action="store_true", help="Share PEFT modules horizontally across dual encoders")
    parser.add_argument("--init_rand", action="store_true",help="If set, PyTorch default init (skip _init_histogram_layer).")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    
    # Few-shot (ShipsEar / DeepShip only)
    parser.add_argument("--k_shot_per_class",type=int,default=0,
        help=("If > 0, restrict the *training* split to at most K segments per class for DeepShip/ShipsEar"),)
    
    # DeepShip only
    parser.add_argument("--deepship_root", type=str, default="datasets/DeepShip")
    parser.add_argument("--deepship_sr", type=int, default=16000)
    parser.add_argument("--deepship_seglen", type=int, default=5)
    parser.add_argument("--deepship_split", choices=["recording", "vessel"], default="recording")
    parser.add_argument("--deepship_split_source", choices=["default", "random", "std_deepship_data_split"], default="default")
    
    # ShipsEar only
    parser.add_argument("--shipsear_root", type=str, default="datasets/ShipsEar")
    parser.add_argument("--shipsear_sr", type=int, default=16000)
    parser.add_argument("--shipsear_seglen", type=int, default=5)
    parser.add_argument("--shipsear_split_strategy", choices=["temporal", "random"], default="temporal",
                    help="'temporal' keeps chronological order, 'random' shuffles recordings per class once and caches to JSON.")

    args = parser.parse_args()
    
    # Validate dual_branch_loss is only used with dual encoder mode
    if getattr(args, "dual_branch_loss", False) and getattr(args, "encoder_mode", "single") != "dual":
        parser.error("--dual_branch_loss can only be used with --encoder_mode dual")

    from transformers import AutoConfig
    def _model_is_local(name: str) -> bool:
        try:
            AutoConfig.from_pretrained(name, local_files_only=True)
            return True
        except Exception:
            return False
    
    model_names_to_check = {args.model_name}
    if getattr(args, "encoder_mode", "single") == "dual":
        if args.model_name_wave is not None:
            model_names_to_check.add(args.model_name_wave)
        if args.model_name_tf is not None:
            model_names_to_check.add(args.model_name_tf)
    
    all_local = all(_model_is_local(name) for name in model_names_to_check)
    if all_local:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print(f"\n[INFO] Models {model_names_to_check} found locally → running in OFFLINE mode.\n")
    else:
        print(f"\n[INFO] Some models in {model_names_to_check} are not cached → network may be used.\n")
    
    # Single runs 
    exp_dir = _group_dir(args) / _build_run_name(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for i in range(args.runs):
        seed = args.seed + i*5
        L.seed_everything(seed, workers=True)
        args.run_idx = i
        dm, label2id, id2label = build_dm_and_labels(args)
        ckpt, metrics = run_single(args, dm, label2id, id2label)
        val_acc  = metrics.get("accuracy", None)
        test_acc = metrics.get("test_accuracy", None)
        if val_acc is not None and test_acc is not None:
            print(f"\nRun {i} best checkpoint: {ckpt}\nSummary: val_acc={val_acc:.4f} | test_acc={test_acc:.4f}")
        elif val_acc is not None:
            print(f"\nRun {i} best checkpoint: {ckpt}\nSummary: val_acc={val_acc:.4f}")
        else:
            print(f"\nRun {i} best checkpoint: {ckpt}")
        metrics_to_save = {k: v for k, v in metrics.items() if not k.startswith("_")}
        print(json.dumps(metrics_to_save, indent=2))
        with open(exp_dir / f"metrics_run{i}.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        all_metrics.append(metrics)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if args.runs > 1:
        aggregate_and_save(all_metrics, exp_dir)

    # ── Aggregated evaluation plots (confusion matrix, ROC, fusion weights) ──
    if args.runs >= 1:
        class_names_for_agg = None
        all_cm_percents = []
        all_roc_data_list = []
        all_run_log_dirs = []
        for m in all_metrics:
            red = m.get("_run_eval_data", {})
            if "cm_percent" in red:
                all_cm_percents.append(red["cm_percent"])
            if "roc_data" in red:
                all_roc_data_list.append(red["roc_data"])
            if "run_log_dir" in red:
                all_run_log_dirs.append(red["run_log_dir"])
        # Reuse id2label from the last run (already available)
        class_names_for_agg = [id2label[i] for i in range(len(id2label))]
        if len(all_cm_percents) > 1 and class_names_for_agg:
            save_aggregated_confusion_matrix(
                all_cm_percents, class_names_for_agg,
                outfile=exp_dir / "confusion_matrix_aggregated.png",
                title=f"{args.dataset.upper()} ({args.runs} runs)",
            )
        if len(all_roc_data_list) > 1 and class_names_for_agg:
            save_aggregated_roc(
                all_roc_data_list, class_names_for_agg,
                outfile=exp_dir / "roc_aggregated.png",
                title=f"{args.dataset.upper()} ({args.runs} runs)",
            )
            save_aggregated_roc_per_class(
                all_roc_data_list, class_names_for_agg,
                outdir=exp_dir,
            )
        # Aggregated fusion weights (dual + choque only)
        if (getattr(args, "encoder_mode", "single") == "dual"
                and len(all_run_log_dirs) > 1):
            save_aggregated_fusion_weights(
                            all_run_log_dirs,
                            outfile=exp_dir / "fusion_weights_aggregated.png",
                            id2label=id2label,
                        )
            # Save aggregated best-epoch fusion weights JSON
            all_fw = [m.get("fusion_weights_best_epoch", {}) for m in all_metrics]
            all_fw = [fw for fw in all_fw if fw and fw.get("type") == "choque"]
            if len(all_fw) > 1:
                g1_arr = np.array([fw["g1_per_class"] for fw in all_fw])
                g2_arr = np.array([fw["g2_per_class"] for fw in all_fw])
                fw_agg = {
                    "type": "choque",
                    "num_runs": len(all_fw),
                    "g1_per_class": {
                        "mean": np.mean(g1_arr, axis=0).tolist(),
                        "std":  np.std(g1_arr, axis=0).tolist(),
                    },
                    "g2_per_class": {
                        "mean": np.mean(g2_arr, axis=0).tolist(),
                        "std":  np.std(g2_arr, axis=0).tolist(),
                    },
                    "class_names": class_names_for_agg,
                }
                fw_json_path = exp_dir / "fusion_weights_best_epoch_aggregated.json"
                with open(fw_json_path, "w") as f:
                    json.dump(fw_agg, f, indent=2)
                print(f"[INFO] Saved aggregated best-epoch fusion weights: {fw_json_path}")

if __name__ == "__main__":
    main()
