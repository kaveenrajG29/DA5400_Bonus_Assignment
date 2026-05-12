from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


@dataclass
class RunConfig:
    data_dir: Path
    output_dir: Path
    config_filename: Path
    dataset: str
    framework: str
    buffer_path: Path
    mode: str
    ipc: int
    spc: int
    Iteration: int
    eval_it: int
    num_eval: int
    epoch_eval_train: int
    batch_train: int
    batch_real: int
    batch_syn: int
    num_experts: int
    train_epochs: int
    save_interval: int
    max_start_epoch: int
    expert_epochs: int
    syn_steps: int
    max_experts: int | None
    max_files: int | None
    load_all: bool
    lr_teacher: float
    lr_feat: float
    lr_lr: float
    mom: float
    l2: float
    net_mom: float
    model: str
    eval_mode: str
    aug: str | None
    inputaug: str
    pix_init: str
    lambda_DM: float
    test_size: float
    seed: int
    representation: str
    max_train_per_class: int | None
    run_full_baseline: bool
    device: str
    synthetic_path: Path | None = None
    comparison_output: Path | None = None
    comparison_table: Path | None = None
    eval_models: str = "MLP,CNNBN,CNNIN,TCN"
    channel: int | None = None
    time_step: int | None = None
    num_classes: int | None = None
    jitter_scale_ratio: float = 0.1
    jitter_ratio: float = 0.01
    max_seg: int = 8
    dual: int = 1
    save_dir: Path | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)


def apply_yaml_config(config: RunConfig) -> None:
    if not config.config_filename.exists():
        return
    with open(config.config_filename, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data_cfg = (raw.get("data") or {}).get(config.dataset)
    if not data_cfg:
        return
    config.channel = int(data_cfg.get("channel", config.channel or 1))
    config.time_step = int(data_cfg.get("time_step", config.time_step or 1))
    config.num_classes = int(data_cfg.get("num_classes", config.num_classes or 1))
    aug_para = data_cfg.get("aug_para") or {}
    config.jitter_scale_ratio = float(aug_para.get("jitter_scale_ratio", config.jitter_scale_ratio))
    config.jitter_ratio = float(aug_para.get("jitter_ratio", config.jitter_ratio))
    config.max_seg = int(aug_para.get("max_seg", config.max_seg))


def config_to_args(config: RunConfig) -> SimpleNamespace:
    d = dict(config.__dict__)
    d["device"] = str(config.device)
    d["inputaug_list"] = config.inputaug.split("_") if config.inputaug else ["raw"]
    d["aug"] = None if config.aug in (None, "None", "none") else config.aug
    d["ipc"] = config.ipc or config.spc
    return SimpleNamespace(**d)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Faithful modular DCDDM time-series condensation pipeline")
    parser.add_argument("--mode", choices=["all", "buffer", "distill"], default="all")
    parser.add_argument("--config_filename", type=Path, default=Path("TimeSeriesCond-master/config.yml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_dcdmm"))
    parser.add_argument("--dataset", type=str, default="local_npz")
    parser.add_argument("--framework", type=str, default="DCDDM")
    parser.add_argument("--buffer_path", type=Path, default=Path("./buffers"))
    parser.add_argument("--ipc", "--spc", dest="ipc", type=int, default=5)
    parser.add_argument("--Iteration", "--epochs", dest="Iteration", type=int, default=200)
    parser.add_argument("--eval_it", type=int, default=50)
    parser.add_argument("--num_eval", type=int, default=2)
    parser.add_argument("--epoch_eval_train", "--eval-epochs", dest="epoch_eval_train", type=int, default=50)
    parser.add_argument("--batch_train", "--batch-size", dest="batch_train", type=int, default=256)
    parser.add_argument("--batch_real", type=int, default=256)
    parser.add_argument("--batch_syn", type=int, default=256)
    parser.add_argument("--num_experts", type=int, default=5)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--max_start_epoch", type=int, default=10)
    parser.add_argument("--expert_epochs", type=int, default=10)
    parser.add_argument("--syn_steps", type=int, default=10)
    parser.add_argument("--max_experts", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--load_all", action="store_true")
    parser.add_argument("--lr_teacher", type=float, default=1e-4)
    parser.add_argument("--lr_feat", "--synth-lr", dest="lr_feat", type=float, default=0.01)
    parser.add_argument("--lr_lr", type=float, default=1e-8)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--net_mom", type=float, default=0.9)
    parser.add_argument("--model", choices=["CNNBN", "CNNIN", "CNNLN", "LSTM", "GRU", "Transformer", "TCN", "ResNet18", "ResNet18BN", "MLP", "cnn", "mlp"], default="CNNIN")
    parser.add_argument("--eval_mode", type=str, default="S")
    parser.add_argument("--aug", type=str, default="None")
    parser.add_argument("--inputaug", type=str, default="raw")
    parser.add_argument("--pix_init", choices=["real", "clustering", "noise"], default="real")
    parser.add_argument("--lambda_DM", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--representation", choices=["magnitude", "real_imag", "real"], default="magnitude")
    parser.add_argument("--max-train-per-class", type=int, default=2000)
    parser.add_argument("--run-full-baseline", action="store_true")
    parser.add_argument("--synthetic-path", type=Path, default=None)
    parser.add_argument("--comparison-output", type=Path, default=None)
    parser.add_argument("--comparison-table", type=Path, default=None)
    parser.add_argument("--eval-models", type=str, default="MLP,CNNBN,CNNIN,TCN")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    d = vars(parser.parse_args())
    d["spc"] = d["ipc"]
    return RunConfig(**d)
