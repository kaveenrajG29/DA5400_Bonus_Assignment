from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from module.config import apply_yaml_config, config_to_args


ACTIVITY_NAMES = {
    "A": "Walk",
    "B": "Run",
    "C": "Jump",
    "D": "Sitting",
    "E": "Empty room",
    "F": "Standing",
    "G": "Wave hands",
    "H": "Clapping",
    "I": "Lay down",
    "J": "Wiping",
    "K": "Squat",
    "L": "Stretching",
}


def activity_name_from_file(path: Path) -> str:
    parts = path.stem.split("_")
    return ACTIVITY_NAMES.get(parts[2], path.stem) if len(parts) >= 3 else path.stem


def load_npz_array(path: Path) -> np.ndarray:
    archive = np.load(path, allow_pickle=False)
    if "data" in archive.files:
        return archive["data"]
    if len(archive.files) == 1:
        return archive[archive.files[0]]
    raise ValueError(f"{path} has multiple arrays {archive.files}; expected one array or key 'data'.")


def complex_to_features(array: np.ndarray, representation: str) -> np.ndarray:
    if np.iscomplexobj(array):
        if representation == "magnitude":
            features = np.abs(array)[:, None, :]
        elif representation == "real_imag":
            features = np.stack([array.real, array.imag], axis=1)
        elif representation == "real":
            features = array.real[:, None, :]
        else:
            raise ValueError(f"Unknown representation: {representation}")
    else:
        features = array[:, None, :] if array.ndim == 2 else array
        if features.ndim == 3 and features.shape[1] > features.shape[2]:
            features = np.swapaxes(features, 1, 2)
    return np.asarray(features, dtype=np.float32)


def load_npz_dataset(config):
    files = sorted(config.data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {config.data_dir}")
    xs, ys, class_names = [], [], []
    for label, path in enumerate(files):
        raw = load_npz_array(path)
        x_class = complex_to_features(raw, config.representation)
        xs.append(x_class)
        ys.append(np.full(x_class.shape[0], label, dtype=np.int64))
        class_names.append(activity_name_from_file(path))
        print(f"Loaded class {label}: {path.name} ({class_names[-1]}) -> {x_class.shape}")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    train_idx, holdout_idx = train_test_split(np.arange(len(y)), test_size=config.test_size, random_state=config.seed, stratify=y)
    val_idx, test_idx = train_test_split(holdout_idx, test_size=0.5, random_state=config.seed, stratify=y[holdout_idx])
    if config.max_train_per_class is not None:
        rng = np.random.default_rng(config.seed)
        keep = []
        for cls in np.unique(y[train_idx]):
            cls_idx = train_idx[y[train_idx] == cls]
            keep.append(rng.choice(cls_idx, size=min(config.max_train_per_class, len(cls_idx)), replace=False))
        train_idx = np.concatenate(keep)
        rng.shuffle(train_idx)
    mean = x[train_idx].mean(axis=(0, 2), keepdims=True)
    std = x[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    x = (x - mean) / std

    def pack(idx):
        return {"samples": torch.tensor(x[idx], dtype=torch.float32), "labels": torch.tensor(y[idx], dtype=torch.long)}

    config.channel = int(x.shape[1])
    config.time_step = int(x.shape[2])
    config.num_classes = int(len(class_names))
    return pack(train_idx), pack(val_idx), pack(test_idx), class_names


def load_torch_dataset(config):
    root = config.data_dir / config.dataset
    train = torch.load(root / "train.pt", map_location="cpu")
    val = torch.load(root / "val.pt", map_location="cpu")
    test = torch.load(root / "test.pt", map_location="cpu")
    merged = {k: torch.cat((train[k], val[k]), dim=0) for k in train.keys()}
    if config.dataset == "epilepsy":
        for part in (merged, val, test):
            part["labels"] = (part["labels"] != 0).long()
    class_names = [str(i) for i in range(int(config.num_classes or merged["labels"].max().item() + 1))]
    return merged, val, test, class_names


def load_data(config):
    apply_yaml_config(config)
    torch_root = config.data_dir / config.dataset
    if torch_root.exists() and (torch_root / "train.pt").exists():
        train, val, test, class_names = load_torch_dataset(config)
    else:
        train, val, test, class_names = load_npz_dataset(config)
    config.channel = int(config.channel or train["samples"].shape[1])
    config.time_step = int(config.time_step or train["samples"].shape[2])
    config.num_classes = int(config.num_classes or len(class_names))
    print(f"Config data: channels={config.channel}, time_step={config.time_step}, classes={config.num_classes}")
    return train, val, test, class_names


def make_loaders(train, val, test, config):
    args = config_to_args(config)
    loaders = {}
    for name, part in {"train": train, "val": val, "test": test}.items():
        ds = TensorDataset(part["samples"].float().to(args.device), part["labels"].long().to(args.device))
        loaders[name] = DataLoader(ds, batch_size=config.batch_train, shuffle=(name == "train"))
        print(f"{name} : X is {part['samples'].shape}, Y is {part['labels'].shape}")
    return loaders
