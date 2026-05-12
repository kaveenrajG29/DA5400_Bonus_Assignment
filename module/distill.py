from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset

from model.dcddm_models import Dualmodel
from model.reparam import ReparamModule, flatten_snapshot
from model.tsmodels import get_network
from module.augmentations import Input_Augmentation, maybe_train_aug
from module.buffer import load_replay_buffers, sample_expert_trajectory
from module.config import config_to_args
from module.data import make_loaders
from module.evaluation import evaluate_synset, get_eval_pool, log_verbose


def student_unroll_steps(student_net, student_params, image_syn, label_syn, syn_lr, args, criterion):
    for _ in range(args.syn_steps):
        grad = None
        for inputaug in args.inputaug_list:
            syn_X_this = Input_Augmentation(image_syn, inputaug)
            syn_X_this = maybe_train_aug(syn_X_this, args, aug=True)
            y_pred, _, _ = student_net(syn_X_this, flat_param=student_params)
            ce_loss = criterion(y_pred, label_syn)
            this_grad = torch.autograd.grad(ce_loss, student_params, create_graph=True, allow_unused=False)[0]
            grad = this_grad if grad is None else grad + this_grad
        student_params = student_params - syn_lr * grad
    return student_params


def trajectory_matching_loss(student_net, expert_trajectory, image_syn, label_syn, syn_lr, args, criterion):
    max_start = min(args.max_start_epoch, len(expert_trajectory) - args.expert_epochs - 1)
    if max_start <= 0:
        raise ValueError("Expert trajectory is shorter than max_start_epoch + expert_epochs.")
    start_epoch = np.random.randint(0, max_start)
    starting_snapshot = expert_trajectory[start_epoch]
    target_snapshot = expert_trajectory[start_epoch + args.expert_epochs]
    target_params = flatten_snapshot(target_snapshot, args.device)
    starting_params = flatten_snapshot(starting_snapshot, args.device)
    student_params = starting_params.detach().clone().requires_grad_(True)
    student_params = student_unroll_steps(student_net, student_params, image_syn, label_syn, syn_lr, args, criterion)
    num_params = student_params.numel()
    param_loss = F.mse_loss(student_params, target_params, reduction="sum") / num_params
    param_dist = F.mse_loss(starting_params, target_params, reduction="sum") / num_params
    grand_loss = param_loss / param_dist.clamp_min(1e-12)
    _, t_emb, f_emb = student_net(image_syn, flat_param=student_params)
    _, t_emb_target, f_emb_target = student_net(image_syn, flat_param=target_params)
    DM_loss_t = F.mse_loss(t_emb_target, t_emb, reduction="mean")
    DM_loss_f = F.mse_loss(f_emb, f_emb_target, reduction="mean")
    total_loss = args.lambda_DM * (DM_loss_t + DM_loss_f) + grand_loss
    return total_loss, grand_loss, DM_loss_t, DM_loss_f


def get_images_clustering(c, n, train, indices_class, args):
    all_images = train["samples"][indices_class[c]].float().to(args.device)
    if all_images.shape[0] <= n:
        return all_images.repeat((math.ceil(n / all_images.shape[0]), 1, 1))[:n]
    temp_model = args.model
    args.model = "CNNBN"
    with torch.no_grad():
        embeddings = get_network(args).to(args.device).embed(all_images).detach().cpu().numpy()
    args.model = temp_model
    km = MiniBatchKMeans(n_clusters=n, batch_size=min(2048, len(embeddings)), n_init="auto", random_state=7)
    km.fit(embeddings)
    centers = torch.tensor(km.cluster_centers_, device=args.device, dtype=torch.float32)
    emb_t = torch.tensor(embeddings, device=args.device, dtype=torch.float32)
    selected = torch.argmin(torch.cdist(centers, emb_t, p=2), dim=1)
    return all_images[selected]


def initialize_synthetic(train, config, args):
    indices_class = [[] for _ in range(args.num_classes)]
    for i in range(len(train["samples"])):
        indices_class[int(train["labels"][i].item())].append(i)
    for c in range(args.num_classes):
        print(f"class c = {c}: {len(indices_class[c])} real images")
    label_syn = torch.arange(args.num_classes, dtype=torch.long, device=args.device).repeat_interleave(args.ipc)
    image_syn = nn.Parameter(torch.empty(args.num_classes * args.ipc, args.channel, args.time_step, device=args.device))
    if config.pix_init in {"real", "clustering"}:
        print("initialize synthetic data from clustering-selected real images")
        for c in range(args.num_classes):
            image_syn.data[c * args.ipc : (c + 1) * args.ipc] = get_images_clustering(c, args.ipc, train, indices_class, args).detach()
    else:
        print("initialize synthetic data from random noise")
        image_syn.data.copy_(torch.randn_like(image_syn))
    return image_syn, label_syn


def distill(config, train, val, test):
    curr_time = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = Path(config.output_dir) / "logged_files" / config.framework / config.dataset / curr_time
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir = save_dir
    args = config_to_args(config)
    args.save_dir = str(save_dir)
    yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}, open(save_dir / "config.yaml", "w", encoding="utf-8"))
    loaders = make_loaders(train, val, test, config)
    eval_it_pool = np.arange(0, config.Iteration + 1, config.eval_it).tolist()
    model_eval_pool = get_eval_pool(config.eval_mode, config.model, config.model)
    log_verbose(args, f"Eval_it_pool : {eval_it_pool}\tmodel_eval_pool : {model_eval_pool}")
    image_syn, label_syn = initialize_synthetic(train, config, args)
    torch.save(copy.deepcopy(image_syn.detach().cpu()), save_dir / "images_init.pt")
    torch.save(label_syn.detach().cpu(), save_dir / "labels_init.pt")
    criterion = nn.CrossEntropyLoss().to(args.device)
    syn_lr = nn.Parameter(torch.tensor(float(config.lr_teacher), device=args.device))
    optimizer_img = torch.optim.SGD([image_syn], lr=config.lr_feat, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=config.lr_lr, momentum=0.5)
    buffer_state = load_replay_buffers(config)
    best_ACC = {m: -1.0 for m in model_eval_pool}
    best_metrics = {}
    time0 = time.time()

    for it in range(config.Iteration + 1):
        save_this_it = False
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                log_verbose(args, f"-------------------------\nEvaluation\nmodel_train = {config.model}, model_eval = {model_eval}, iteration = {it}")
                metrics_runs = {}
                loaders["syn"] = DataLoader(TensorDataset(image_syn.detach().cpu(), label_syn.detach().cpu()), batch_size=config.batch_train, shuffle=True)
                for it_eval in range(config.num_eval):
                    net_eval = Dualmodel(args).to(args.device)
                    _, metrics = evaluate_synset(it_eval, net_eval, loaders, args, use_data="val", aug=True)
                    for k, v in metrics.items():
                        metrics_runs.setdefault(k, []).append(v)
                for k, vals in metrics_runs.items():
                    arr = np.asarray(vals, dtype=float)
                    log_verbose(args, f"Eval {k}: {np.nanmean(arr):.5f} +/- {np.nanstd(arr):.5f}")
                acc_mean = float(np.nanmean(metrics_runs["Accuracy"]))
                if acc_mean > best_ACC[model_eval]:
                    best_ACC[model_eval] = acc_mean
                    save_this_it = True
                    test_runs = {}
                    for it_eval in range(config.num_eval):
                        net_eval = Dualmodel(args).to(args.device)
                        _, metrics = evaluate_synset(it_eval, net_eval, loaders, args, use_data="test", aug=True)
                        for k, v in metrics.items():
                            test_runs.setdefault(k, []).append(v)
                    log_verbose(args, "This is the best model so far!")
                    best_metrics = {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))} for k, v in test_runs.items()}
                    for k, stat in best_metrics.items():
                        log_verbose(args, f"Test {k}: {stat['mean']:.5f} +/- {stat['std']:.5f}")
        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            torch.save(image_syn.detach().cpu(), save_dir / f"images_{it}.pt")
            torch.save(label_syn.detach().cpu(), save_dir / f"labels_{it}.pt")
            if save_this_it:
                torch.save(image_syn.detach().cpu(), save_dir / "images_best.pt")
                torch.save(label_syn.detach().cpu(), save_dir / "labels_best.pt")
                log_verbose(args, f"Synthetic Training Epoch : {it}, Saved Best!")

        student_net = ReparamModule(Dualmodel(args).to(args.device))
        student_net.train()
        expert_trajectory = sample_expert_trajectory(buffer_state, config)
        total_loss, grand_loss, dm_t, dm_f = trajectory_matching_loss(student_net, expert_trajectory, image_syn, label_syn, syn_lr, args, criterion)
        optimizer_img.zero_grad(set_to_none=True)
        optimizer_lr.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer_img.step()
        optimizer_lr.step()
        args.lr_teacher = syn_lr.detach()
        if it % 10 == 0:
            log_verbose(args, f"DEBUG : syn_lr: {syn_lr.item():.8f}")
            log_verbose(args, f"Synthetic Training Epoch : {it}/{config.Iteration}, Matching_loss : {grand_loss.item():.4f}, DM_loss_t  : {dm_t.item():.4f}, DM_loss_f : {dm_f.item():.4f}, cost {time.time()-time0:.4f}s")
            time0 = time.time()

    np.savez_compressed(save_dir / "condensed_dataset.npz", x=image_syn.detach().cpu().numpy(), y=label_syn.detach().cpu().numpy())
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_test": best_metrics, "best_val_accuracy": best_ACC}, f, indent=2)
    return image_syn.detach().cpu().numpy(), label_syn.detach().cpu().numpy(), best_metrics
