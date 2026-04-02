"""
DeepShip DataModule and dataset utilities.
This module prepares a segment-level dataset from the DeepShip corpus and
exposes PyTorch Lightning data loaders.
Supports encoder modes:
- single: one encoder (AST or waveform)
- dual: wave encoder (HF extractor) + TF encoder (AST uses custom mel spec for identical framing)
- spectrum: frequency bands (custom mel spec)
"""
from __future__ import annotations
import json
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
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
# Configuration and constants
# ──────────────────────────────────────────────────────────────────────────────
CLASSES = ["Cargo", "Passengership", "Tanker", "Tug"]
SEED = 42
random.seed(SEED)
THIS_DIR: Path = Path(__file__).resolve().parent
DATA_ROOT: Path = THIS_DIR / "DeepShip"
SPLIT_FILE: Path = THIS_DIR / "deepship_dataset_split.json"
STD_SPLIT_DIR: Path = THIS_DIR / "std_deepship_data_split"
_DEBUG_COUNTS: Dict[str, Dict[str, Tuple[int, int]]] = {}
_SKIP_STATS: Dict[str, Dict[str, int]] = {
    cls: {
        "unpaired_rec": 0,
        "missing_seg_dir": 0,
        "few_segments": 0,
        "skipped_segs": 0,
    }
    for cls in CLASSES
}
LETTER_RANK = {ch: idx + 1 for idx, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}

def _folder_key(p: Path) -> Tuple[str, int, int]:
    m = re.match(r"(?P<date>\d{8})(?P<letter>[a-z]?)-(?P<idx>\d+)$", p.name)
    if not m:
        return ("00000000", 99, 999_999)
    date, letter, idx = m.group("date"), m.group("letter"), int(m.group("idx"))
    return (date, LETTER_RANK.get(letter, 0), idx)

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("deepship.datamodule")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Segmentation utilities
# ──────────────────────────────────────────────────────────────────────────────
def _process_file_task(args: Tuple[Path, Path, str, int, int]):
    file_path, seg_root, ship, target_sr_, seg_len_s = args
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    if sr != target_sr_:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr_)
    duration = audio.shape[0]
    segment_len = target_sr_ * seg_len_s
    n_segments = math.ceil(duration / segment_len)
    written = 0
    for i in range(n_segments):
        start = int(i * segment_len)
        end = int(min(start + segment_len, duration))
        if end - start != segment_len:
            continue
        clip = audio[start:end]
        subfolder_name = file_path.parent.name
        file_stem = file_path.stem
        out_dir = seg_root / ship / subfolder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / f"{file_stem}_{ship}-Segment_{i + 1}.wav"
        sf.write(out_fp, clip, samplerate=target_sr_)
        written += 1
    return written

def _generate_segments(
    dataset_dir: Path,
    segments_dir: Path,
    target_sr: int = 16000,
    segment_length_s: int = 5,
    num_workers: int = 4,
) -> None:
    ship_types = CLASSES
    tasks: List[Tuple[Path, Path, str, int, int]] = []
    for ship in ship_types:
        folder_path = dataset_dir / ship
        seg_root = segments_dir
        for subfolder in sorted(folder_path.iterdir()):
            if not subfolder.is_dir():
                continue
            for wav in sorted(subfolder.glob("*.wav")):
                tasks.append((wav, seg_root, ship, target_sr, segment_length_s))
    logger.info(
        "Starting segmentation: %d files → %s (sr=%d, len=%ds, workers=%d)",
        len(tasks), segments_dir, target_sr, segment_length_s, num_workers,
    )
    if not tasks:
        logger.warning("No WAV files found under %s", dataset_dir)
        return
    with Pool(processes=num_workers) as pool:
        results = pool.map(_process_file_task, tasks)
    total_written = sum(results)
    logger.info("Segmentation complete: %d segments written", total_written)

def _ensure_segments_exist(
    data_root: Path,
    seg_root: Path,
    sample_rate: int,
    segment_length_s: int,
    num_workers: int,
) -> None:
    if seg_root.exists():
        logger.info("Segments directory exists: %s — skipping generation.", seg_root)
        return
    logger.info("Creating segments at %s", seg_root)
    seg_root.mkdir(parents=True, exist_ok=True)
    _generate_segments(
        dataset_dir=data_root,
        segments_dir=seg_root,
        target_sr=sample_rate,
        segment_length_s=segment_length_s,
        num_workers=num_workers,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Metadata scanning and pairing
# ──────────────────────────────────────────────────────────────────────────────
def _assert_dirs(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    missing = [c for c in CLASSES if not (data_root / c).exists()]
    if missing:
        raise FileNotFoundError(f"Missing class folders: {missing}")

def _load_meta_lines(fp: Path) -> List[str]:
    if not fp.exists():
        return []
    return [ln.strip() for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]

def _map_recordings_to_meta(
    cls_name: str, rec_dirs: List[Path], meta_lines: List[str]
) -> Dict[str, Dict]:
    rec_by_date: Dict[str, List[Path]] = defaultdict(list)
    for p in rec_dirs:
        rec_by_date[_folder_key(p)[0]].append(p)
    for lst in rec_by_date.values():
        lst.sort(key=_folder_key)
    meta_by_date: Dict[str, List[str]] = defaultdict(list)
    for ln in meta_lines:
        parts = ln.split(",")
        if len(parts) >= 5:
            meta_by_date[parts[3]].append(ln)
    for lst in meta_by_date.values():
        lst.sort(key=lambda ln: int(ln.split(",")[4]))
    vessels: Dict[str, Dict] = {}
    for date, rec_list in rec_by_date.items():
        meta_list = meta_by_date.get(date, [])
        n_pair = min(len(rec_list), len(meta_list))
        for k in range(n_pair):
            rec, meta_ln = rec_list[k], meta_list[k]
            vessel = meta_ln.split(",")[2].strip()
            vessels.setdefault(vessel, {"rec_folders": [], "meta": meta_ln})
            vessels[vessel]["rec_folders"].append(rec)
        _SKIP_STATS[cls_name]["unpaired_rec"] += max(0, len(rec_list) - n_pair)
    return vessels

def _scan_metadata_and_group(data_root: Path, seg_root: Path) -> Dict[str, Dict[str, Dict]]:
    _assert_dirs(data_root)
    grouped: Dict[str, Dict[str, Dict]] = {}
    for cls in CLASSES:
        cls_dir = data_root / cls
        meta_fp = cls_dir / f"{cls.lower()}-metafile"
        rec_dirs = sorted([p for p in cls_dir.iterdir() if p.is_dir()])
        meta_ln = _load_meta_lines(meta_fp)
        vessels = _map_recordings_to_meta(cls, rec_dirs, meta_ln)
        qualified: Dict[str, Dict] = {}
        for v, info in vessels.items():
            q_recs: List[Path] = []
            q_segs = 0
            for r in info["rec_folders"]:
                seg_dir = seg_root / cls / r.name
                if not seg_dir.exists():
                    _SKIP_STATS[cls]["missing_seg_dir"] += 1
                    continue
                wavs = list(seg_dir.glob("*.wav"))
                if len(wavs) < 4:
                    _SKIP_STATS[cls]["few_segments"] += 1
                    _SKIP_STATS[cls]["skipped_segs"] += len(wavs)
                    continue
                q_recs.append(r)
                q_segs += len(wavs)
            if q_recs:
                qualified[v] = {"rec_folders": q_recs, "meta": info["meta"]}
                _DEBUG_COUNTS.setdefault(cls, {})[v] = (len(q_recs), q_segs)
        grouped[cls] = qualified
    return grouped

# ──────────────────────────────────────────────────────────────────────────────
# Split utilities
# ──────────────────────────────────────────────────────────────────────────────
def _split_list(items: List, ratios=(0.7, 0.15, 0.15), temporal: bool = True) -> Tuple[List, List, List]:
    """Split items. If temporal=True, preserve order; otherwise shuffle."""
    if not temporal:
        random.shuffle(items)
    n = len(items)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return items[:n_train], items[n_train: n_train + n_val], items[n_train + n_val:]

def _split_list_temporal(items: List, ratios=(0.7, 0.15, 0.15)) -> Tuple[List, List, List]:
    """Split maintaining order (no shuffle) for temporal splits."""
    n = len(items)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return items[:n_train], items[n_train: n_train + n_val], items[n_train + n_val:]

@dataclass
class SplitConfig:
    split_level: str = "recording"
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_source: str = "default"

def _parse_std_split_line(ln: str) -> Tuple[str, str]:
    parts = Path(ln).as_posix().split("/")
    if len(parts) < 4 or parts[0] != "DeepShip":
        raise ValueError(f"Unexpected std split line format: {ln}")
    cls = parts[1]
    rec = parts[2]
    if cls not in CLASSES:
        raise ValueError(f"Unknown class '{cls}' in std split line: {ln}")
    return cls, rec

def _generate_or_load_std_txt_splits(
    seg_root: Path,
    sample_rate: int,
    segment_length_s: int,
    seed: int = SEED,
) -> Dict[str, Dict[str, List[str]]]:
    param_suffix = f"std_deepship_data_split__recording__{segment_length_s}s_{sample_rate}hz.json"
    split_fp = SPLIT_FILE.with_name(param_suffix)
    if split_fp.exists():
        logger.info("Loading existing std_deepship_data_split splits: %s", split_fp)
        return json.loads(split_fp.read_text())
    train_fp = STD_SPLIT_DIR / "train.txt"
    test_fp = STD_SPLIT_DIR / "test.txt"
    if not train_fp.exists() or not test_fp.exists():
        raise FileNotFoundError(f"Missing std split files. Expected:\n  {train_fp}\n  {test_fp}")
    train_lines = _load_meta_lines(train_fp)
    test_lines = _load_meta_lines(test_fp)
    train_pool: Dict[str, set] = {c: set() for c in CLASSES}
    test_pool: Dict[str, set] = {c: set() for c in CLASSES}
    for ln in train_lines:
        cls, rec = _parse_std_split_line(ln)
        train_pool[cls].add(rec)
    for ln in test_lines:
        cls, rec = _parse_std_split_line(ln)
        test_pool[cls].add(rec)
    rng = random.Random(seed)
    split_dict: Dict[str, Dict[str, List[str]]] = {}
    for cls in CLASSES:
        train_recs = sorted(train_pool[cls])
        rng.shuffle(train_recs)
        n_val = int(0.2 * len(train_recs))
        val_recs = sorted(train_recs[:n_val])
        tr_recs = sorted(train_recs[n_val:])
        te_recs = sorted(test_pool[cls])
        split_dict[cls] = {"train": tr_recs, "val": val_recs, "test": te_recs}
    split_fp.write_text(json.dumps(split_dict, indent=2))
    logger.info("Wrote std_deepship_data_split splits to %s", split_fp)
    return split_dict

def _generate_or_load_splits(
    grouped: Dict[str, Dict[str, Dict]],
    split_cfg: SplitConfig,
    seg_root: Path,
    sample_rate: int,
    segment_length_s: int,
) -> Dict[str, Dict[str, List]]:
    if split_cfg.split_source == "std_deepship_data_split":
        if split_cfg.split_level != "recording":
            raise ValueError("std_deepship_data_split only supported with split_level='recording'.")
        return _generate_or_load_std_txt_splits(
            seg_root=seg_root, sample_rate=sample_rate, segment_length_s=segment_length_s, seed=SEED
        )
    param_suffix = f"deepship__{split_cfg.split_source}__{split_cfg.split_level}_split_{segment_length_s}s_{sample_rate}hz.json"
    split_fp = SPLIT_FILE.with_name(param_suffix)

    if split_fp.exists():
        logger.info("Loading existing splits: %s", split_fp)
        return json.loads(split_fp.read_text())
    temporal = (split_cfg.split_source != "random")
    logger.info("Generating new %s-level %s splits", split_cfg.split_level, "TEMPORAL" if temporal else "RANDOM")
    split_dict: Dict[str, Dict[str, List]] = {}
    if split_cfg.split_level == "vessel":
        for cls, vmap in grouped.items():
            def vessel_sort_key(v):
                recs = vmap[v]["rec_folders"]
                if not recs:
                    return ("99999999", 99, 999999)
                return min(_folder_key(r) for r in recs)
            vessels = sorted(vmap.keys(), key=vessel_sort_key)
            tr, va, te = _split_list(vessels, split_cfg.ratios, temporal=temporal)
            split_dict[cls] = {"train": tr, "val": va, "test": te}
    else:  # recording level
        for cls, vmap in grouped.items():
            all_recs: List[Tuple[Tuple, str]] = []
            for v in vmap.keys():
                for rec in vmap[v]["rec_folders"]:
                    all_recs.append((_folder_key(rec), rec.name))
            all_recs.sort(key=lambda x: x[0])
            rec_names = [name for _, name in all_recs]
            tr, va, te = _split_list(rec_names, split_cfg.ratios, temporal=temporal)
            split_dict[cls] = {"train": tr, "val": va, "test": te}
    split_fp.write_text(json.dumps(split_dict, indent=2))
    logger.info("Wrote deterministic %s splits to %s", "TEMPORAL" if temporal else "RANDOM", split_fp)
    return split_dict

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class DSAudioDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        class_to_idx: Dict[str, int],
        sample_rate: int,
        segment_length_s: int,
        expose_vessel: bool = False,
        vessel_to_idx: Optional[Dict[str, int]] = None,
        transform=None,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.segment_length_s = segment_length_s
        self.transform = transform
        self.expose_vessel = expose_vessel
        self.vessel_to_idx = vessel_to_idx or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.samples[idx]
        file_path = Path(entry["file_path"])
        wav, sr = sf.read(file_path, always_2d=False)
        if sr != self.sample_rate:
            raise ValueError(f"Unexpected sample rate {sr} in {file_path}; expected {self.sample_rate}.")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).to(torch.float32)
        if self.transform:
            wav_t = self.transform(wav_t)
        sample = {
            "waveform": wav_t,
            "class": self.class_to_idx[entry["class"]],
            "recording": entry["recording"],
        }
        if self.expose_vessel:
            sample["vessel"] = self.vessel_to_idx[entry["vessel"]]
        return sample

# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DSAudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: int = 16000,
        segment_length_s: int = 5,
        split_level: str = "recording",
        split_source: str = "default",
        data_root: Path = DATA_ROOT,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        k_shot_per_class: int = 0,
        encoder_mode: str = "single",
        model_name_wave: Optional[str] = None,
        model_name_tf: Optional[str] = None,
        augment: bool = True,
        ast_input_type: str = "logmel",
        mfcc_n_mfcc: int = 32,
        mfcc_n_mels: int = 128,
        cqt_n_bins: int = 84,
        norm_type: str = "sample",
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.cqt_n_bins = cqt_n_bins
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.segment_length_s = segment_length_s
        self.split_cfg = SplitConfig(split_level=split_level, split_source=split_source)
        self.data_root = data_root
        self.seg_root = self._segments_root()
        self.persistent_workers = (num_workers > 0)
        self.prefetch_factor = 4
        self.k_shot_per_class = int(k_shot_per_class)
        self.encoder_mode = encoder_mode

        if ast_input_type not in ("logmel", "mfcc", "cqt", "vqt"):
            raise ValueError(f"Unknown ast_input_type: {ast_input_type}. Use 'logmel', 'mfcc', 'cqt', or 'vqt'.")
        self.ast_input_type = ast_input_type
        self.mfcc_n_mfcc = mfcc_n_mfcc
        self.mfcc_n_mels = mfcc_n_mels

        # Flags / holders
        self._is_ast_model = False          # single mode
        self._tf_is_ast = False             # dual mode TF branch
        self.mel_transform_single = None
        self.mel_transform_tf = None
        self.mfcc_transform_single = None
        self.mfcc_transform_tf = None

        self.waveform_augmentation = get_waveform_augmentation(
            enabled=augment,
            noise_prob=0.5,
            noise_snr_db_range=(15.0, 25.0),
        )

        # Populated in prepare_data/setup
        self.grouped: Dict[str, Dict[str, Dict]] = {}
        self.splits: Dict[str, Dict[str, List]] = {}
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
        self.vessel_to_idx: Dict[str, int] = {}
        self.train_set: Optional[DSAudioDataset] = None
        self.val_set: Optional[DSAudioDataset] = None
        self.test_set: Optional[DSAudioDataset] = None
        self.label2id = {c: i for i, c in enumerate(CLASSES)}
        self.id2label = {i: c for c, i in self.label2id.items()}

        # ────────────────────────────────────────────────────────────────
        # Feature extraction setup per encoder_mode
        # ────────────────────────────────────────────────────────────────
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
                logger.info(
                    f"[DeepShip] Single(AST): n_fft={self.n_fft}, hop_length={self.hop_length}, "
                    f"time_frames={self.actual_time_frames}"
                )
                if self.ast_input_type == "mfcc":
                    logger.info(f"  MFCC config: n_mfcc={self.mfcc_n_mfcc}, n_mels={self.mfcc_n_mels}")
            else:
                self.fe = AutoFeatureExtractor.from_pretrained(model_name, do_normalize=True)
                logger.info("[DeepShip] Single(waveform): using AutoFeatureExtractor")

        elif self.encoder_mode == "dual":
            wave_name = model_name_wave or model_name
            tf_name = model_name_tf or model_name
            # wave branch extractor (always HF)
            self.fe_wave = AutoFeatureExtractor.from_pretrained(wave_name, do_normalize=True)
            # TF branch: if AST => use custom mel spec (identical framing)
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
                logger.info(
                    f"[DeepShip] Dual: TF is AST => custom mel spec. "
                    f"n_fft={self.n_fft}, hop_length={self.hop_length}, time_frames={self.actual_time_frames}"
                )
                if self.ast_input_type == "mfcc":
                    logger.info(f"  MFCC config: n_mfcc={self.mfcc_n_mfcc}, n_mels={self.mfcc_n_mels}")
            else:
                self.fe_tf = AutoFeatureExtractor.from_pretrained(tf_name, do_normalize=True)
                logger.info("[DeepShip] Dual: TF is not AST => AutoFeatureExtractor for TF")

        else:
            raise ValueError(f"Unknown encoder_mode: {self.encoder_mode}")

        # K-shot bookkeeping
        if self.k_shot_per_class > 0:
            suffix = (
                f"kshot_k{self.k_shot_per_class}_"
                f"{self.split_cfg.split_level}_"
                f"{self.segment_length_s}s_{self.sample_rate}hz.json"
            )
            self.kshot_fp = SPLIT_FILE.with_name(suffix)
        else:
            self.kshot_fp = None

        # Augmentation
        self.augment_enabled = augment
        self.training = True
        self.augmentation = get_ship_augmentation(
            enabled=augment,
            noise_prob=0.5,
            noise_snr_db_range=(15.0, 25.0),
            time_mask_prob=0.5,
            time_mask_max_ratio=0.15,
            freq_mask_prob=0.5,
            freq_mask_max_ratio=0.10,
        )
        if self.augmentation is not None:
            logger.info("[DeepShip] Audio augmentation ENABLED (noise + time/freq masking)")
        else:
            logger.info("[DeepShip] Audio augmentation DISABLED")

    def _segments_root(self) -> Path:
        return self.data_root / f"Segments_{self.segment_length_s}s_{self.sample_rate}hz"

    def prepare_data(self) -> None:
        _ensure_segments_exist(
            data_root=self.data_root,
            seg_root=self.seg_root,
            sample_rate=self.sample_rate,
            segment_length_s=self.segment_length_s,
            num_workers=self.num_workers,
        )
        if self.split_cfg.split_source == "std_deepship_data_split":
            self.grouped = {}
            self.splits = _generate_or_load_splits(
                grouped={},
                split_cfg=self.split_cfg,
                seg_root=self.seg_root,
                sample_rate=self.sample_rate,
                segment_length_s=self.segment_length_s,
            )
        else:
            self.grouped = _scan_metadata_and_group(self.data_root, self.seg_root)
            self.splits = _generate_or_load_splits(
                self.grouped, self.split_cfg, self.seg_root, self.sample_rate, self.segment_length_s
            )

    def _print_counts(self) -> None:
        from collections import Counter

        def per_class(ds):
            # ds.samples entries contain "class" as string
            return Counter([s["class"] for s in ds.samples])

        n_tr, n_va, n_te = len(self.train_set), len(self.val_set), len(self.test_set)
        print("\n=== DeepShip segment samples per split ===")
        print(f"train: {n_tr}\nval  : {n_va}\ntest : {n_te}\nall  : {n_tr + n_va + n_te}")

        tr_c, va_c, te_c = per_class(self.train_set), per_class(self.val_set), per_class(self.test_set)
        hdr = "{:<15s} {:>8s} {:>8s} {:>8s} {:>8s}"
        row = "{:<15s} {:>8d} {:>8d} {:>8d} {:>8d}"
        print("\nPer-class segment counts:")
        print(hdr.format("CLS", "train", "val", "test", "total"))
        for c in CLASSES:
            tot = tr_c.get(c, 0) + va_c.get(c, 0) + te_c.get(c, 0)
            print(row.format(c, tr_c.get(c, 0), va_c.get(c, 0), te_c.get(c, 0), tot))
        print()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_set is not None:
            return
        all_vessels = sorted({v for cls_map in self.grouped.values() for v in cls_map.keys()}) if self.grouped else []
        self.vessel_to_idx = {v: i for i, v in enumerate(all_vessels)}

        def _collect_samples(split: str) -> List[Dict]:
            lst: List[Dict] = []
            for cls in CLASSES:
                if cls not in self.splits:
                    continue
                if self.split_cfg.split_level == "vessel":
                    for v in self.splits[cls][split]:
                        for rec in self.grouped[cls][v]["rec_folders"]:
                            seg_dir = self.seg_root / cls / rec.name
                            for wav_fp in seg_dir.glob("*.wav"):
                                lst.append(
                                    {
                                        "file_path": str(wav_fp),
                                        "class": cls,
                                        "vessel": v,
                                        "recording": rec.name,
                                        "split": split,
                                    }
                                )
                else:
                    for rec_name in self.splits[cls][split]:
                        seg_dir = self.seg_root / cls / rec_name
                        for wav_fp in seg_dir.glob("*.wav"):
                            vessel = ""
                            if self.grouped and cls in self.grouped:
                                for v, info in self.grouped[cls].items():
                                    if any(r.name == rec_name for r in info["rec_folders"]):
                                        vessel = v
                                        break
                            lst.append(
                                {
                                    "file_path": str(wav_fp),
                                    "class": cls,
                                    "vessel": vessel,
                                    "recording": rec_name,
                                    "split": split,
                                }
                            )
            return lst

        expose_vessel = self.split_cfg.split_level == "vessel"
        train_samples = _collect_samples("train")
        val_samples = _collect_samples("val")
        test_samples = _collect_samples("test")
        if self.k_shot_per_class > 0:
            train_samples = self._apply_k_shot(train_samples)
        self.train_set = DSAudioDataset(
            train_samples, self.class_to_idx, self.sample_rate, self.segment_length_s,
            expose_vessel=expose_vessel, vessel_to_idx=self.vessel_to_idx
        )
        self.val_set = DSAudioDataset(
            val_samples, self.class_to_idx, self.sample_rate, self.segment_length_s,
            expose_vessel=expose_vessel, vessel_to_idx=self.vessel_to_idx
        )
        self.test_set = DSAudioDataset(
            test_samples, self.class_to_idx, self.sample_rate, self.segment_length_s,
            expose_vessel=expose_vessel, vessel_to_idx=self.vessel_to_idx
        )
        assert len(self.train_set) > 0
        assert len(self.val_set) > 0
        assert len(self.test_set) > 0
        self._print_counts()

    def _apply_k_shot(self, samples: List[Dict]) -> List[Dict]:
        k = self.k_shot_per_class
        assert self.kshot_fp is not None
        if self.kshot_fp.exists():
            logger.info("Loading k-shot train subset from %s", self.kshot_fp)
            payload = json.loads(self.kshot_fp.read_text())
            return payload.get("samples", [])
        logger.info("Creating k-shot (k=%d) subset for DeepShip train split…", k)
        from collections import defaultdict
        by_cls_rec: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        for s in samples:
            cls = s["class"]
            rec = s["recording"]
            by_cls_rec[cls][rec].append(s)
        selected: List[Dict] = []
        for cls, rec_map in by_cls_rec.items():
            total_segs = sum(len(v) for v in rec_map.values())
            target = min(k, total_segs)
            if target == total_segs:
                for rec_name, lst in rec_map.items():
                    selected.extend(lst)
                continue
            rec_names = list(rec_map.keys())
            random.shuffle(rec_names)
            for rn in rec_names:
                random.shuffle(rec_map[rn])
            cls_selected: List[Dict] = []
            while len(cls_selected) < target:
                progressed = False
                for rn in rec_names:
                    lst = rec_map[rn]
                    if not lst:
                        continue
                    cls_selected.append(lst.pop())
                    progressed = True
                    if len(cls_selected) >= target:
                        break
                if not progressed:
                    break
            selected.extend(cls_selected)
        payload = {"k_per_class": k, "split_level": self.split_cfg.split_level, "samples": selected}
        self.kshot_fp.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote k-shot subset to %s (total %d samples)", self.kshot_fp, len(selected))
        return selected
    
    # ────────────────────────────────────────────────────────────────────────
    # Collate
    # ────────────────────────────────────────────────────────────────────────
    def collate_fn(self, batch):
        waves = [b["waveform"].numpy() if hasattr(b["waveform"], "numpy") else b["waveform"] for b in batch]
        labels = [b["class"] for b in batch]

        # Apply waveform augmentation for wave encoders (when training)
        if self.waveform_augmentation is not None and self.training:
            waves_tensor = torch.stack([torch.from_numpy(w) if isinstance(w, np.ndarray) else w for w in waves])
            waves_tensor = self.waveform_augmentation(waves_tensor)
            waves = [w.numpy() for w in waves_tensor]

        # -------------------- single --------------------
        if self.encoder_mode == "single":
            if self._is_ast_model:
                if self.ast_input_type == "mfcc":
                    x = create_fullband_mfcc_batched(
                        waveforms=waves,
                        mfcc_transform=self.mfcc_transform_single,
                        norm_type=self.norm_type,
                    )
                                    
                elif self.ast_input_type == "cqt":
                    x = create_fullband_cqt_batched(
                        waveforms=waves, 
                        cqt_transform=self.cqt_transform_single, 
                        norm_type=self.norm_type,
                    )
                elif self.ast_input_type == "vqt":
                    x = create_fullband_vqt_batched(
                        waveforms=waves, 
                        vqt_transform=self.vqt_transform_single, 
                        norm_type=self.norm_type,
                    )

                else:
                    x = create_fullband_spectrogram_batched(
                        waveforms=waves,
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
                    out = self.fe(waves, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
                if "input_features" in out:
                    x = out["input_features"]
                    if self.augmentation is not None and self.training:
                        x = self.augmentation(x)
                    return {"input_features": x, "labels": torch.tensor(labels, dtype=torch.long)}
                if "input_values" in out:
                    return {"input_values": out["input_values"], "labels": torch.tensor(labels, dtype=torch.long)}
                raise RuntimeError("Unknown feature extractor output structure (single mode).")

        # -------------------- dual --------------------
        if self.encoder_mode == "dual":
            batch_dict = {"labels": torch.tensor(labels, dtype=torch.long)}

            # wave branch (HF)
            with torch.no_grad():
                out_wave = self.fe_wave(waves, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
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
                        waveforms=waves,
                        mfcc_transform=self.mfcc_transform_tf,
                        norm_type=self.norm_type,
                    )
                elif self.ast_input_type == "cqt":
                    tf_x = create_fullband_cqt_batched(
                        waveforms=waves,
                        cqt_transform=self.cqt_transform_tf,
                        norm_type=self.norm_type,
                    )
                elif self.ast_input_type == "vqt":
                    tf_x = create_fullband_vqt_batched(
                        waveforms=waves,
                        vqt_transform=self.vqt_transform_tf,
                        norm_type=self.norm_type,
                    )
                else:
                    tf_x = create_fullband_spectrogram_batched(
                        waveforms=waves,
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
                    out_tf = self.fe_tf(waves, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
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
