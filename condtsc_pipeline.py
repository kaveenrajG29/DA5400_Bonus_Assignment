from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


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


@dataclass
class RunConfig:
    data_dir: Path
    output_dir: Path
    spc: int
    epochs: int
    batch_size: int
    eval_epochs: int
    n_syn: int
    m_real: int
    inner_lr: float
    synth_lr: float
    lambda_emb: float
    test_size: float
    seed: int
    representation: str
    model: str
    max_train_per_class: int | None
    run_full_baseline: bool
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_array(path: Path) -> np.ndarray:
    archive = np.load(path, allow_pickle=False)
    if "data" in archive.files:
        return archive["data"]
    if len(archive.files) == 1:
        return archive[archive.files[0]]
    raise ValueError(f"{path} has multiple arrays {archive.files}; pass files with one array or key 'data'.")


def complex_to_features(array: np.ndarray, representation: str) -> np.ndarray:
    """Convert input to float32 shape [samples, channels, length]."""
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


def load_dataset(config: RunConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(config.data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {config.data_dir}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    class_names: list[str] = []
    for label, path in enumerate(files):
        raw = load_npz_array(path)
        x_class = complex_to_features(raw, config.representation)
        xs.append(x_class)
        ys.append(np.full(x_class.shape[0], label, dtype=np.int64))
        class_name = activity_name_from_file(path)
        class_names.append(class_name)
        print(f"Loaded class {label}: {path.name} ({class_name}) -> {x_class.shape}")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y, class_names


def activity_name_from_file(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 3:
        code = parts[2]
        return ACTIVITY_NAMES.get(code, path.stem)
    return path.stem


def stratified_split(
    x: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def standardize(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def limit_per_class(x: np.ndarray, y: np.ndarray, max_per_class: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_per_class is None:
        return x, y
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        size = min(max_per_class, len(cls_idx))
        keep.append(rng.choice(cls_idx, size=size, replace=False))
    keep_idx = np.concatenate(keep)
    rng.shuffle(keep_idx)
    return x[keep_idx], y[keep_idx]


def init_synthetic_kmeans(x_train: np.ndarray, y_train: np.ndarray, spc: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    synth_x = []
    synth_y = []
    flat = x_train.reshape(x_train.shape[0], -1)
    for cls in np.unique(y_train):
        cls_flat = flat[y_train == cls]
        n_clusters = min(spc, len(cls_flat))
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(2048, len(cls_flat)),
            n_init="auto",
            random_state=seed,
        )
        centers = km.fit(cls_flat).cluster_centers_.reshape(n_clusters, *x_train.shape[1:])
        if n_clusters < spc:
            pad = np.repeat(centers[-1:], spc - n_clusters, axis=0)
            centers = np.concatenate([centers, pad], axis=0)
        synth_x.append(centers.astype(np.float32))
        synth_y.append(np.full(spc, cls, dtype=np.int64))
    return np.concatenate(synth_x, axis=0), np.concatenate(synth_y, axis=0)


class ConvClassifier(nn.Module):
    def __init__(self, channels: int, length: int, classes: int, hidden: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=7, padding=3),
            nn.GroupNorm(8, hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden, classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).squeeze(-1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.embed(x)
        return embedding if return_embedding else self.classifier(embedding)


class MLPClassifier(nn.Module):
    def __init__(self, channels: int, length: int, classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * length, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden, classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.embed(x)
        return embedding if return_embedding else self.classifier(embedding)


def build_model(name: str, channels: int, length: int, classes: int) -> nn.Module:
    if name == "cnn":
        return ConvClassifier(channels, length, classes)
    if name == "mlp":
        return MLPClassifier(channels, length, classes)
    raise ValueError(f"Unknown model: {name}")


def tensor_dict(module: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().clone()) for name, value in module.named_parameters())


def functional_forward(module: nn.Module, params: OrderedDict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return functional_call(module, params, (x,))


def model_embedding(module: nn.Module, params: OrderedDict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return functional_call(module, params, (x,), {"return_embedding": True}, tie_weights=False)


def train_params(
    module: nn.Module,
    params: OrderedDict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    create_graph: bool,
) -> OrderedDict[str, torch.Tensor]:
    current = OrderedDict((k, v.clone().requires_grad_(True)) for k, v in params.items())
    for _ in range(steps):
        logits = functional_forward(module, current, x)
        loss = F.cross_entropy(logits, y)
        grads = torch.autograd.grad(loss, tuple(current.values()), create_graph=create_graph)
        updated = OrderedDict((k, value - lr * grad) for (k, value), grad in zip(current.items(), grads))
        if not create_graph:
            updated = OrderedDict((k, value.detach().requires_grad_(True)) for k, value in updated.items())
        current = updated
    return current


def frequency_view(x: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft(x, dim=-1)
    return torch.log1p(torch.abs(spectrum))


def low_pass(x: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
    spectrum = torch.fft.fft(x, dim=-1)
    keep = max(1, int(x.shape[-1] * keep_ratio / 2))
    mask = torch.zeros(x.shape[-1], device=x.device, dtype=torch.bool)
    mask[:keep] = True
    mask[-keep:] = True
    filtered = torch.where(mask.view(*([1] * (x.ndim - 1)), -1), spectrum, torch.zeros_like(spectrum))
    return torch.fft.ifft(filtered, dim=-1).real


def phase_perturb(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    spectrum = torch.fft.fft(x, dim=-1)
    magnitude = torch.abs(spectrum)
    phase = torch.angle(spectrum) + torch.randn_like(x) * std
    return torch.fft.ifft(magnitude * torch.exp(1j * phase), dim=-1).real


def magnitude_perturb(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    spectrum = torch.fft.fft(x, dim=-1)
    magnitude = torch.abs(spectrum) * (1.0 + torch.randn_like(x) * std)
    phase = torch.angle(spectrum)
    return torch.fft.ifft(magnitude.clamp_min(0.0) * torch.exp(1j * phase), dim=-1).real


def augmented_views(synth_x: torch.Tensor) -> list[torch.Tensor]:
    raw = synth_x
    lpf = low_pass(raw)
    pp = phase_perturb(lpf)
    mp = magnitude_perturb(pp)
    return [raw, lpf, pp, mp]


def sample_real_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device)
    return x[idx], y[idx]


def normalized_param_mse(a: OrderedDict[str, torch.Tensor], b: OrderedDict[str, torch.Tensor], base: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    numerator = sum(torch.sum((a[k] - b[k]) ** 2) for k in a)
    denominator = sum(torch.sum((base[k] - b[k]) ** 2) for k in a).clamp_min(1e-8)
    return numerator / denominator


def condense(config: RunConfig, x_train: np.ndarray, y_train: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(config.device)
    channels, length = x_train.shape[1], x_train.shape[2]
    init_x, init_y = init_synthetic_kmeans(x_train, y_train, config.spc, config.seed)

    real_x = torch.tensor(x_train, dtype=torch.float32, device=device)
    real_y = torch.tensor(y_train, dtype=torch.long, device=device)
    synth_x = nn.Parameter(torch.tensor(init_x, dtype=torch.float32, device=device))
    synth_y = torch.tensor(init_y, dtype=torch.long, device=device)
    optimizer = torch.optim.SGD([synth_x], lr=config.synth_lr)

    for epoch in trange(config.epochs, desc="Condensing"):
        module_t = build_model(config.model, channels, length, classes).to(device)
        module_f = build_model(config.model, channels, length, classes).to(device)
        base_t = tensor_dict(module_t)
        base_f = tensor_dict(module_f)
        real_batch_x, real_batch_y = sample_real_batch(real_x, real_y, config.batch_size)
        real_batch_f = frequency_view(real_batch_x)

        total_loss = torch.tensor(0.0, device=device)
        for view in augmented_views(synth_x):
            synth_f = frequency_view(view)

            syn_t = train_params(module_t, base_t, view, synth_y, config.n_syn, config.inner_lr, create_graph=True)
            syn_f = train_params(module_f, base_f, synth_f, synth_y, config.n_syn, config.inner_lr, create_graph=True)
            with torch.enable_grad():
                real_t = train_params(module_t, base_t, real_batch_x, real_batch_y, config.m_real, config.inner_lr, create_graph=False)
                real_f = train_params(module_f, base_f, real_batch_f, real_batch_y, config.m_real, config.inner_lr, create_graph=False)

            grad_loss = normalized_param_mse(syn_t, real_t, base_t) + normalized_param_mse(syn_f, real_f, base_f)
            emb_t = (model_embedding(module_t, syn_t, view) - model_embedding(module_t, real_t, view)).mean(dim=0).pow(2).mean()
            emb_f = (model_embedding(module_f, syn_f, synth_f) - model_embedding(module_f, real_f, synth_f)).mean(dim=0).pow(2).mean()
            total_loss = total_loss + grad_loss + config.lambda_emb * (emb_t + emb_f)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if epoch % max(1, config.epochs // 10) == 0 or epoch == config.epochs - 1:
            print(f"epoch={epoch:04d} loss={total_loss.item():.4f}")

    return synth_x.detach().cpu().numpy(), synth_y.detach().cpu().numpy()


def train_eval_arrays(
    config: RunConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: int,
    description: str,
) -> float:
    device = torch.device(config.device)
    channels, length = train_x.shape[1], train_x.shape[2]
    model = build_model(config.model, channels, length, classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)),
        batch_size=min(config.batch_size, len(train_y)),
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=config.batch_size,
        shuffle=False,
    )

    for _ in trange(config.eval_epochs, desc=description):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            logits = model(bx.to(device))
            pred = logits.argmax(dim=1).cpu()
            correct += int((pred == by).sum())
            total += int(by.numel())
    return correct / total


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="CondTSC-style time-series condensation pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--spc", type=int, default=10, help="Synthetic samples per class")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-epochs", type=int, default=100)
    parser.add_argument("--n-syn", type=int, default=3, help="Inner synthetic training steps")
    parser.add_argument("--m-real", type=int, default=5, help="Inner real-data training steps")
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--synth-lr", type=float, default=1.0)
    parser.add_argument("--lambda-emb", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation", choices=["magnitude", "real_imag", "real"], default="magnitude")
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--max-train-per-class", type=int, default=2000)
    parser.add_argument("--run-full-baseline", action="store_true", help="Also train/evaluate on the capped full training set")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return RunConfig(**vars(args))


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    x, y, class_names = load_dataset(config)
    x_train, y_train, x_test, y_test = stratified_split(x, y, config.test_size, config.seed)
    x_train, y_train = limit_per_class(x_train, y_train, config.max_train_per_class, config.seed)
    x_train, x_test, mean, std = standardize(x_train, x_test)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}, Classes: {class_names}")

    synth_x, synth_y = condense(config, x_train, y_train, len(class_names))
    condensed_accuracy = train_eval_arrays(
        config, synth_x, synth_y, x_test, y_test, len(class_names), "Evaluating condensed data"
    )
    print(f"Condensed-data test accuracy: {condensed_accuracy:.4f}")

    metrics = {"condensed_test_accuracy": condensed_accuracy}
    if config.run_full_baseline:
        full_accuracy = train_eval_arrays(
            config, x_train, y_train, x_test, y_test, len(class_names), "Evaluating full/capped data"
        )
        metrics["full_or_capped_test_accuracy"] = full_accuracy
        print(f"Full/capped-data test accuracy: {full_accuracy:.4f}")

    np.savez_compressed(
        config.output_dir / "condensed_dataset.npz",
        x=synth_x,
        y=synth_y,
        class_names=np.array(class_names),
        mean=mean,
        std=std,
        representation=np.array(config.representation),
    )
    with open(config.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}, f, indent=2)
    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
