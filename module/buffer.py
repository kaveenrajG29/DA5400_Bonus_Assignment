from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from model.dcddm_models import Dualmodel
from module.config import config_to_args
from module.data import make_loaders
from module.evaluation import epoch, epoch_origin


def dcddm_buffer_dir(config) -> Path:
    aug = config.aug if config.aug is not None else "None"
    return Path(config.buffer_path) / config.framework / config.dataset / aug / config.model


def save_replay_buffers(trajectories, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    while (save_dir / f"replay_buffer_{n}.pt").exists():
        n += 1
    path = save_dir / f"replay_buffer_{n}.pt"
    torch.save(trajectories, path)
    print(f"Saving {path}")
    return path


def generate_expert_trajectories(config, train, val, test):
    args = config_to_args(config)
    loaders = make_loaders(train, val, test, config)
    criterion = nn.CrossEntropyLoss().to(args.device)
    save_dir = dcddm_buffer_dir(config)
    trajectories = []
    for it in range(config.num_experts):
        print("======================")
        print(f"Trajectory ID : {it}")
        teacher_net = Dualmodel(args).to(args.device)
        print(f"model count is : {sum(p.numel() for p in teacher_net.parameters())}\tfinal_rep is : {teacher_net.final_rep}")
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=config.lr_teacher, momentum=config.mom, weight_decay=config.l2)
        timestamps = [[p.detach().cpu() for p in teacher_net.parameters()]]
        time0 = time.time()
        for e in range(config.train_epochs):
            train_loss, train_metrics = epoch("train", {"data_loader": loaders["train"], "model": config.model}, teacher_net, teacher_optim, criterion, args, aug=True)
            test_loss, test_metrics = epoch_origin("test", {"data_loader": loaders["test"], "model": config.model}, teacher_net, None, criterion, args, aug=False)
            print(f"Epoch: {e}\tTrain Loss : {train_loss:.6f}\t" + ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
            print(f"Epoch: {e}\tTest Loss : {test_loss:.6f}\t" + ", ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))
            print(f"Cost : {time.time()-time0:.3f}s")
            time0 = time.time()
            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
        trajectories.append(timestamps)
        if len(trajectories) == config.save_interval:
            save_replay_buffers(trajectories, save_dir)
            trajectories = []
    if trajectories:
        save_replay_buffers(trajectories, save_dir)
    return load_replay_buffers(config)


def load_replay_buffers(config):
    expert_dir = dcddm_buffer_dir(config)
    expert_files = []
    n = 0
    while (expert_dir / f"replay_buffer_{n}.pt").exists():
        expert_files.append(expert_dir / f"replay_buffer_{n}.pt")
        n += 1
    if not expert_files:
        raise AssertionError(f"No buffers detected at {expert_dir}")
    if config.max_files is not None:
        expert_files = expert_files[: config.max_files]
    if config.load_all:
        buffer = []
        for path in expert_files:
            buffer += torch.load(path, map_location="cpu")
        if config.max_experts is not None:
            buffer = buffer[: config.max_experts]
        random.shuffle(buffer)
        return {"load_all": True, "buffer": buffer, "files": expert_files}
    random.shuffle(expert_files)
    buffer = torch.load(expert_files[0], map_location="cpu")
    if config.max_experts is not None:
        buffer = buffer[: config.max_experts]
    random.shuffle(buffer)
    return {"load_all": False, "buffer": buffer, "files": expert_files, "file_idx": 0, "expert_idx": 0}


def sample_expert_trajectory(state, config):
    if state["load_all"]:
        return state["buffer"][random.randrange(len(state["buffer"]))]
    trajectory = state["buffer"][state["expert_idx"]]
    state["expert_idx"] += 1
    if state["expert_idx"] == len(state["buffer"]):
        state["expert_idx"] = 0
        state["file_idx"] = (state["file_idx"] + 1) % len(state["files"])
        if state["file_idx"] == 0:
            random.shuffle(state["files"])
        if config.max_files != 1:
            state["buffer"] = torch.load(state["files"][state["file_idx"]], map_location="cpu")
        if config.max_experts is not None:
            state["buffer"] = state["buffer"][: config.max_experts]
        random.shuffle(state["buffer"])
    return trajectory
