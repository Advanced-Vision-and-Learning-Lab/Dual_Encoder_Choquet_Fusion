# datasets/ShipsEarDataModule.py
"""
ShipsEar DataModule with support for single, dual, and spectrum-encoder modes.
"""
from __future__ import annotations
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import lightning as L
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoConfig
from src.audio_augmentation import get_ship_augmentation, get_waveform_augmentation
from src.audio_filtering import (
    create_fullband_mel_transform,
    create_fullband_spectrogram_batched,
    create_fullband_mfcc_transform,
    create_fullband_mfcc_batched,
    create_fullband_cqt_batched,
    create_fullband_vqt_batched,
    create_fullband_cqt_transform,
    create_fullband_vqt_transform,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants / config
# ──────────────────────────────────────────────────────────────────────────────
CLASSES = ["A", "B", "C", "D", "E"]
THIS_DIR: Path = Path(__file__).resolve().parent
DATA_ROOT: Path = THIS_DIR / "ShipsEar"
SPLIT_FILE_BASENAME = "shipsear_splits_{seglen}s_{sr}hz.json"
RANDOM_SPLIT_FILE_BASENAME = "random_shipsear_split_{seglen}s_{sr}hz.json"
GROUPS = {
    "A": ["Fishboat", "Trawler", "Mussel boat", "Tugboat", "Dredger"],
    "B": ["Motorboat", "Pilot ship", "Sailboat"],
    "C": ["Passengers"],
    "D": ["Ocean liner", "RORO"],
    "E": ["Natural ambient noise"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Segmentation (only if missing)
# ──────────────────────────────────────────────────────────────────────────────
def _generate_segments_if_needed(dataset_dir: Path, target_sr: int, segment_length: int) -> None:
    excel_fp = dataset_dir / "shipsEar.xlsx"
    already_segmented = True
    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            already_segmented = False
            break
        any_wavs = False
        for sub in cls_dir.iterdir() if cls_dir.exists() else []:
            if sub.is_dir() and any(str(p).lower().endswith(".wav") for p in sub.glob("*.wav")):
                any_wavs = True
                break
        if not any_wavs:
            already_segmented = False
            break
    if already_segmented:
        print("[ShipsEar] Segments found; skipping segmentation.")
        return
    if not excel_fp.exists():
        raise FileNotFoundError(
            f"[ShipsEar] shipsEar.xlsx not found at {excel_fp}. Cannot generate segments."
        )
    print("[ShipsEar] Segments not found; running one-time segmentation…")
    _generate_segments(dataset_dir, target_sr=target_sr, segment_length=segment_length)
    print("[ShipsEar] Segmentation complete.")

def _generate_segments(dataset_dir: Path, target_sr: int = 16000, segment_length: int = 5) -> None:
    import librosa
    import soundfile as sf  # local alias
    data_csv = pd.read_excel(dataset_dir / "shipsEar.xlsx")
    ship_type_groups = GROUPS
    ship_keys = CLASSES
    segmentation_needed = False
    for ship in ship_keys:
        segments_path = str(dataset_dir / ship)
        temp_samples = data_csv[data_csv["Type"].isin(ship_type_groups[ship])]
        for _, row in temp_samples.iterrows():
            subfolder_name = str(row["Filename"]).split(".")[0]
            segment_folder_path = Path(segments_path) / subfolder_name
            if not segment_folder_path.exists():
                segmentation_needed = True
                break
        if segmentation_needed:
            break
    if not segmentation_needed:
        print("[ShipsEar] All files already segmented. Skipping segmentation.")
        return
    for ship in ship_keys:
        segments_path = dataset_dir / ship
        temp_samples = data_csv[data_csv["Type"].isin(ship_type_groups[ship])]
        for _, row in temp_samples.iterrows():
            subfolder_name = str(row["Filename"]).split(".")[0]
            file_name = str(row["Filename"])
            segment_folder_path = segments_path / subfolder_name
            if segment_folder_path.exists():
                continue
            file_path = dataset_dir / file_name
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            seg_len = int(target_sr * segment_length)
            n_segments = int(math.ceil(len(audio) / seg_len))
            for i in range(n_segments):
                start_i = i * seg_len
                end_i = min(start_i + seg_len, len(audio))
                if (end_i - start_i) != seg_len:
                    continue
                out = audio[start_i:end_i]
                segment_folder_path.mkdir(parents=True, exist_ok=True)
                out_fp = segment_folder_path / f"{Path(file_name).stem}-Segment_{i+1}.wav"
                sf.write(out_fp, out, samplerate=target_sr)

def _shipsear_folder_key(p: Path) -> Tuple[int, str]:
    """
    Parse ShipsEar folder names like:
    - '15__10_07_13_radaUno_Pasa'  -> (15, '10_07_13')
    - '85__E__1L'                   -> (85, 'E')
    - '91__E__7H_N'                 -> (91, 'E')
    Returns (index, date_or_class) for chronological sorting by index.
    """
    name = p.name
    match = re.match(r"^(\d+)__(.*)$", name)
    if match:
        idx = int(match.group(1))
        rest = match.group(2)
        return (idx, rest)
    return (999999, name)

# ──────────────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────────────
def _make_or_load_splits(
    dataset_dir: Path,
    split_json_fp: Path,
    ratios=(0.7, 0.15, 0.15),
    split_strategy: str = "temporal",
) -> Dict[str, Dict[str, List[str]]]:
    if split_json_fp.exists():
        print(f"[ShipsEar] Loading existing split JSON: {split_json_fp}")
        return json.loads(split_json_fp.read_text())
    if split_strategy not in ("temporal", "random"):
        raise ValueError(f"Unknown split_strategy: {split_strategy}. Use 'temporal' or 'random'.")
    if split_strategy == "temporal":
        print(f"[ShipsEar] Creating per-class TEMPORAL splits → {split_json_fp.name}")
    else:
        print(f"[ShipsEar] Creating per-class RANDOM (recording-level) splits → {split_json_fp.name}")
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    splits = {"train": [], "val": [], "test": []}
    for c in CLASSES:
        c_dir = dataset_dir / c
        if not c_dir.exists():
            raise FileNotFoundError(f"[ShipsEar] Missing class folder: {c_dir}")
        recs = [p for p in c_dir.iterdir() if p.is_dir()]
        if split_strategy == "temporal":
            recs.sort(key=_shipsear_folder_key)
        else:
            # Keep recording-level split (no leakage), but randomize which recordings go to which split
            random.shuffle(recs)
        n = len(recs)
        n_train = int(ratios[0] * n)
        n_val = int(ratios[1] * n)
        train_recs = recs[:n_train]
        val_recs = recs[n_train:n_train + n_val]
        test_recs = recs[n_train + n_val:]
        for p in train_recs:
            splits["train"].append([f"{c}/{p.name}", class_to_idx[c]])
        for p in val_recs:
            splits["val"].append([f"{c}/{p.name}", class_to_idx[c]])
        for p in test_recs:
            splits["test"].append([f"{c}/{p.name}", class_to_idx[c]])
    payload = {"classes": class_to_idx, "splits": splits}
    split_json_fp.write_text(json.dumps(payload, indent=2))
    if split_strategy == "temporal":
        print(f"[ShipsEar] Wrote TEMPORAL split JSON: {split_json_fp}")
    else:
        print(f"[ShipsEar] Wrote RANDOM split JSON: {split_json_fp}")
    return payload

# ──────────────────────────────────────────────────────────────────────────────
# Dataset / DataModule
# ──────────────────────────────────────────────────────────────────────────────
class _ShipsEarDataset(Dataset):
    def __init__(self, segment_tuples: List[Tuple[str, int]]):
        self.items = segment_tuples

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_fp, cls_id = self.items[idx]
        wav, sr = sf.read(wav_fp, always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).to(torch.float32)
        return wav_t, int(cls_id)

class ShipsEarDataModule(L.LightningDataModule):
    def __init__(
        self,
        shipsear_root: Path = DATA_ROOT,
        batch_size: int = 16,
        num_workers: int = 4,
        sample_rate: int = 16000,
        segment_length_s: int = 5,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        split_json_name: Optional[str] = None,
        k_shot_per_class: int = 0,
        encoder_mode: str = "single",
        model_name_wave: Optional[str] = None,
        model_name_tf: Optional[str] = None,
        augment: bool = True,
        split_strategy: str = "temporal",
        ast_input_type: str = "logmel",
        mfcc_n_mfcc: int = 32,
        mfcc_n_mels: int = 128,
        cqt_n_bins: int = 84,
        norm_type: str = "sample",
        run_idx: int = 0,
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.cqt_n_bins = cqt_n_bins
        self.root = Path(shipsear_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.segment_length_s = segment_length_s
        self.k_shot_per_class = int(k_shot_per_class)
        self.persistent_workers = (num_workers > 0)
        self.prefetch_factor = 4
        self.encoder_mode = encoder_mode
        if split_strategy not in ("temporal", "random"):
            raise ValueError(f"Unknown split_strategy: {split_strategy}. Use 'temporal' or 'random'.")
        self.split_strategy = split_strategy

        if ast_input_type not in ("logmel", "mfcc", "cqt", "vqt"):
            raise ValueError(f"Unknown ast_input_type: {ast_input_type}. Use 'logmel', 'mfcc', 'cqt', or 'vqt'.")
        self.ast_input_type = ast_input_type
        self.mfcc_n_mfcc = mfcc_n_mfcc
        self.mfcc_n_mels = mfcc_n_mels

        # Flags / holders
        self._is_ast_model = False          # single mode
        self._tf_is_ast = False             # dual TF branch
        self.mel_transform_single = None
        self.mel_transform_tf = None
        self.mfcc_transform_single = None
        self.mfcc_transform_tf = None

        self.waveform_augmentation = get_waveform_augmentation(
            enabled=augment,
            noise_prob=0.5,
            noise_snr_db_range=(15.0, 30.0),
        )

        # ────────────────────────────────────────────────────────────────────
        # Feature extraction setup per encoder_mode
        # ────────────────────────────────────────────────────────────────────
        if self.encoder_mode == "single":
            cfg = AutoConfig.from_pretrained(model_name)
            self._is_ast_model = (cfg.model_type == "audio-spectrogram-transformer")
            if self._is_ast_model:
                self.n_fft = 4096
                self.hop_length = 512
                self.n_mels = 64
                self.actual_time_frames = (self.sample_rate * self.segment_length_s) // self.hop_length + 1
                self.mel_transform_single = create_fullband_mel_transform(
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                )
                self.mfcc_transform_single = create_fullband_mfcc_transform(
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.mfcc_n_mels,
                    n_mfcc=self.mfcc_n_mfcc,
                )

                self.cqt_transform_single = create_fullband_cqt_transform(
                    sr=self.sample_rate, hop_length=self.hop_length, n_bins=self.cqt_n_bins
                )
                self.vqt_transform_single = create_fullband_vqt_transform(
                    sr=self.sample_rate, hop_length=self.hop_length, n_bins=self.cqt_n_bins
                )

                self.fe = None
                print(
                    f"[ShipsEar] Single(AST): n_fft={self.n_fft}, hop_length={self.hop_length}, "
                    f"time_frames={self.actual_time_frames}"
                )
                if self.ast_input_type == "mfcc":
                    print(f"  MFCC config: n_mfcc={self.mfcc_n_mfcc}, n_mels={self.mfcc_n_mels}")
            else:
                self.fe = AutoFeatureExtractor.from_pretrained(model_name, do_normalize=True)
                print("[ShipsEar] Single(waveform): using AutoFeatureExtractor")

        elif self.encoder_mode == "dual":
            wave_name = model_name_wave or model_name
            tf_name = model_name_tf or model_name
            self.fe_wave = AutoFeatureExtractor.from_pretrained(wave_name, do_normalize=True)
            tf_cfg = AutoConfig.from_pretrained(tf_name)
            self._tf_is_ast = (tf_cfg.model_type == "audio-spectrogram-transformer")
            if self._tf_is_ast:
                self.n_fft = 4096
                self.hop_length = 512
                self.n_mels = 64
                self.actual_time_frames = (self.sample_rate * self.segment_length_s) // self.hop_length + 1
                self.mel_transform_tf = create_fullband_mel_transform(
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                )
                self.mfcc_transform_tf = create_fullband_mfcc_transform(
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.mfcc_n_mels,
                    n_mfcc=self.mfcc_n_mfcc,
                )

                self.cqt_transform_tf = create_fullband_cqt_transform(
                    sr=self.sample_rate, hop_length=self.hop_length, n_bins=self.cqt_n_bins
                )
                self.vqt_transform_tf = create_fullband_vqt_transform(
                    sr=self.sample_rate, hop_length=self.hop_length, n_bins=self.cqt_n_bins
                )

                self.fe_tf = None
                print(
                    f"[ShipsEar] Dual: TF is AST => custom mel spec. "
                    f"n_fft={self.n_fft}, hop_length={self.hop_length}, time_frames={self.actual_time_frames}"
                )
                if self.ast_input_type == "mfcc":
                    print(f"  MFCC config: n_mfcc={self.mfcc_n_mfcc}, n_mels={self.mfcc_n_mels}")
            else:
                self.fe_tf = AutoFeatureExtractor.from_pretrained(tf_name, do_normalize=True)
                print("[ShipsEar] Dual: TF is not AST => AutoFeatureExtractor for TF")

        else:
            raise ValueError(f"Unknown encoder_mode: {self.encoder_mode}")

        # Labels
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.id2label = {i: c for c, i in self.class_to_idx.items()}
        self.label2id = {c: i for i, c in self.id2label.items()}

        # Augmentation
        self.augment_enabled = augment
        self.training = True
        self.augmentation = get_ship_augmentation(
            enabled=augment,
            noise_prob=0.5,
            noise_snr_db_range=(15.0, 30.0),
            time_mask_prob=0.5,
            time_mask_max_ratio=0.20,
            freq_mask_prob=0.5,
            freq_mask_max_ratio=0.10,
        )
        print(
            "[ShipsEar] Audio augmentation ENABLED"
            if self.augmentation is not None
            else "[ShipsEar] Audio augmentation DISABLED"
        )

        # Split file path
        if split_json_name is None:
            if self.split_strategy == "random":
                split_json_name = RANDOM_SPLIT_FILE_BASENAME.format(
                    seglen=self.segment_length_s, sr=self.sample_rate
                )
            else:
                split_json_name = SPLIT_FILE_BASENAME.format(
                    seglen=self.segment_length_s, sr=self.sample_rate
                )
        self.split_json_fp = THIS_DIR / split_json_name
        self.run_idx = run_idx
        if self.k_shot_per_class > 0:
            self.kshot_json_fp = THIS_DIR / (
                f"shipsear_kshot_k{self.k_shot_per_class}_{self.segment_length_s}s_{self.sample_rate}hz_run{self.run_idx}.json"
            )
        else:
            self.kshot_json_fp = None
        self._splits_payload: Optional[Dict] = None
        self.train_set: Optional[_ShipsEarDataset] = None
        self.val_set: Optional[_ShipsEarDataset] = None
        self.test_set: Optional[_ShipsEarDataset] = None

    def prepare_data(self) -> None:
        _generate_segments_if_needed(self.root, self.sample_rate, self.segment_length_s)
        self._splits_payload = _make_or_load_splits(
            self.root,
            self.split_json_fp,
            ratios=(0.7, 0.15, 0.15),
            split_strategy=self.split_strategy,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_set is not None:
            return
        assert self._splits_payload is not None, "prepare_data() must be called before setup()."

        def expand(split_name: str) -> List[Tuple[str, int]]:
            tuples: List[Tuple[str, int]] = []
            for rel_rec_path, cls_id in self._splits_payload["splits"][split_name]:
                rec_dir = (self.root / rel_rec_path).resolve()
                for wav_fp in sorted(rec_dir.glob("*.wav")):
                    tuples.append((str(wav_fp), int(cls_id)))
            return tuples

        train_items = expand("train")
        val_items = expand("val")
        test_items = expand("test")
        if self.k_shot_per_class > 0:
            train_items = self._apply_k_shot(train_items)
        self.train_set = _ShipsEarDataset(train_items)
        self.val_set = _ShipsEarDataset(val_items)
        self.test_set = _ShipsEarDataset(test_items)
        assert len(self.train_set) > 0, "Empty train set."
        assert len(self.val_set) > 0, "Empty val set."
        assert len(self.test_set) > 0, "Empty test set."
        self._print_counts()

    def _print_counts(self) -> None:
        def per_class(ds: _ShipsEarDataset) -> Dict[int, int]:
            d: Dict[int, int] = {i: 0 for i in range(len(CLASSES))}
            for _, cid in ds.items:
                d[cid] += 1
            return d

        n_tr, n_va, n_te = len(self.train_set), len(self.val_set), len(self.test_set)
        print("\n=== ShipsEar segment samples per split ===")
        print(f"train: {n_tr}\nval  : {n_va}\ntest : {n_te}\nall  : {n_tr + n_va + n_te}")

        tr_c, va_c, te_c = per_class(self.train_set), per_class(self.val_set), per_class(self.test_set)
        hdr = "{:<4s} {:>8s} {:>8s} {:>8s} {:>8s}"
        row = "{:<4s} {:>8d} {:>8d} {:>8d} {:>8d}"
        print("\nPer-class segment counts:")
        print(hdr.format("CLS", "train", "val", "test", "total"))
        for c in CLASSES:
            i = self.class_to_idx[c]
            tot = tr_c[i] + va_c[i] + te_c[i]
            print(row.format(c, tr_c[i], va_c[i], te_c[i], tot))
        print()

    def _apply_k_shot(self, train_items: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        k = self.k_shot_per_class
        assert self.kshot_json_fp is not None
        if self.kshot_json_fp.exists():
            print(f"[ShipsEar] Loading k-shot train subset from {self.kshot_json_fp}")
            payload = json.loads(self.kshot_json_fp.read_text())
            return [(p, int(cid)) for p, cid in payload.get("train_items", [])]
        print(f"[ShipsEar] Creating k-shot (k={k}) subset for train split…")
        from collections import defaultdict
        by_cls_rec: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for wav_fp, cls_id in train_items:
            rec_name = Path(wav_fp).parent.name
            by_cls_rec[int(cls_id)][rec_name].append(wav_fp)
        selected: List[Tuple[str, int]] = []
        for cls_id, rec_map in by_cls_rec.items():
            total_segs = sum(len(v) for v in rec_map.values())
            target = min(k, total_segs)
            if target == total_segs:
                for paths in rec_map.values():
                    for p in paths:
                        selected.append((p, cls_id))
                continue
            rec_names = list(rec_map.keys())
            random.shuffle(rec_names)
            for rn in rec_names:
                random.shuffle(rec_map[rn])
            cls_selected: List[Tuple[str, int]] = []
            while len(cls_selected) < target:
                progressed = False
                for rn in rec_names:
                    paths = rec_map[rn]
                    if not paths:
                        continue
                    cls_selected.append((paths.pop(), cls_id))
                    progressed = True
                    if len(cls_selected) >= target:
                        break
                if not progressed:
                    break
            selected.extend(cls_selected)
        payload = {"k_per_class": k, "train_items": [[p, int(cid)] for (p, cid) in selected]}
        self.kshot_json_fp.write_text(json.dumps(payload, indent=2))
        print(f"[ShipsEar] Wrote k-shot subset to {self.kshot_json_fp}")
        return selected
    
    # ────────────────────────────────────────────────────────────────────────
    # Collate
    # ────────────────────────────────────────────────────────────────────────
    def collate_fn(self, batch):
        waves, labels = zip(*batch)
        waves_np = [w.numpy() if hasattr(w, "numpy") else w for w in waves]

        # Apply waveform augmentation for wave encoders (when training)
        if self.waveform_augmentation is not None and self.training:
            waves_tensor = torch.stack([torch.from_numpy(w) if isinstance(w, np.ndarray) else w for w in waves_np])
            waves_tensor = self.waveform_augmentation(waves_tensor)
            waves_np = [w.numpy() for w in waves_tensor]

        # -------- single --------
        if self.encoder_mode == "single":
            if self._is_ast_model:
                if self.ast_input_type == "mfcc":
                    x = create_fullband_mfcc_batched(
                        waveforms=waves_np,
                        mfcc_transform=self.mfcc_transform_single,
                        norm_type=self.norm_type,
                    )

                elif self.ast_input_type == "cqt":
                    x = create_fullband_cqt_batched(
                        waveforms=waves_np, 
                        cqt_transform=self.cqt_transform_single, 
                        norm_type=self.norm_type,
                    )
                elif self.ast_input_type == "vqt":
                    x = create_fullband_vqt_batched(
                        waveforms=waves_np, 
                        vqt_transform=self.vqt_transform_single, 
                        norm_type=self.norm_type,
                    )

                else:
                    x = create_fullband_spectrogram_batched(
                        waveforms=waves_np,
                        mel_transform=self.mel_transform_single,
                        norm_type=self.norm_type,
                    )
                if self.augmentation is not None and self.training:
                    x = x.transpose(1, 2)  # [B,time,bins] -> [B,bins,time]
                    x = self.augmentation(x)
                    x = x.transpose(1, 2)
                return {"input_values": x, "labels": torch.tensor(labels, dtype=torch.long)}
            else:
                with torch.no_grad():
                    out = self.fe(waves_np, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
                if "input_features" in out:
                    x = out["input_features"]
                    if self.augmentation is not None and self.training:
                        x = self.augmentation(x)
                    return {"input_features": x, "labels": torch.tensor(labels, dtype=torch.long)}
                if "input_values" in out:
                    return {"input_values": out["input_values"], "labels": torch.tensor(labels, dtype=torch.long)}
                raise RuntimeError("Unknown feature extractor output structure (single mode).")

        # -------- dual --------
        if self.encoder_mode == "dual":
            batch_dict = {"labels": torch.tensor(labels, dtype=torch.long)}

            # wave branch (HF)
            with torch.no_grad():
                out_wave = self.fe_wave(waves_np, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            if "input_features" in out_wave:
                wave_x = out_wave["input_features"]
                if self.augmentation is not None and self.training:
                    wave_x = self.augmentation(wave_x)
                batch_dict["wave_input_features"] = wave_x
            if "input_values" in out_wave:
                batch_dict["wave_input_values"] = out_wave["input_values"]

            # tf branch
            if self._tf_is_ast:
                if self.ast_input_type == "mfcc":
                    tf_x = create_fullband_mfcc_batched(
                        waveforms=waves_np,
                        mfcc_transform=self.mfcc_transform_tf,
                        norm_type=self.norm_type,
                    )
                elif self.ast_input_type == "cqt":
                    tf_x = create_fullband_cqt_batched(
                        waveforms=waves_np,
                        cqt_transform=self.cqt_transform_tf,
                        norm_type=self.norm_type,
                    )
                elif self.ast_input_type == "vqt":
                    tf_x = create_fullband_vqt_batched(
                        waveforms=waves_np,
                        vqt_transform=self.vqt_transform_tf,
                        norm_type=self.norm_type,
                    )
                else:
                    tf_x = create_fullband_spectrogram_batched(
                        waveforms=waves_np,
                        mel_transform=self.mel_transform_tf,
                        norm_type=self.norm_type,
                    )
                if self.augmentation is not None and self.training:
                    tf_x = tf_x.transpose(1, 2)  # [B,time,bins] -> [B,bins,time]
                    tf_x = self.augmentation(tf_x)
                    tf_x = tf_x.transpose(1, 2)
                batch_dict["tf_input_values"] = tf_x
            else:
                with torch.no_grad():
                    out_tf = self.fe_tf(waves_np, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
                if "input_features" in out_tf:
                    tf_x = out_tf["input_features"]
                    if self.augmentation is not None and self.training:
                        tf_x = self.augmentation(tf_x)
                    batch_dict["tf_input_features"] = tf_x
                if "input_values" in out_tf:
                    batch_dict["tf_input_values"] = out_tf["input_values"]
            return batch_dict

        raise ValueError(f"Unknown encoder_mode: {self.encoder_mode}")

    # ────────────────────────────────────────────────────────────────────────
    # Dataloaders
    # ────────────────────────────────────────────────────────────────────────
    def train_dataloader(self):
        self.training = True
        if self.augmentation is not None:
            self.augmentation.train()
        kwargs = dict(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        if self.num_workers > 0:
            kwargs.update(dict(persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor))
        return DataLoader(**kwargs)

    def val_dataloader(self):
        self.training = False
        if self.augmentation is not None:
            self.augmentation.eval()
        kwargs = dict(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        if self.num_workers > 0:
            kwargs.update(dict(persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor))
        return DataLoader(**kwargs)

    def test_dataloader(self):
        self.training = False
        if self.augmentation is not None:
            self.augmentation.eval()
        kwargs = dict(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        if self.num_workers > 0:
            kwargs.update(dict(persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor))
        return DataLoader(**kwargs)
