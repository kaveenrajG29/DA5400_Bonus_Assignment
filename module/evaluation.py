from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from module.augmentations import Input_Augmentation, maybe_train_aug


def log_verbose(args, msg):
    print(msg)
    if getattr(args, "save_dir", None):
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.save_dir) / "log.txt", "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def logits_from_output(out):
    return out[0] if isinstance(out, tuple) else out


def metric_dict_from_arrays(y_true, prob, num_classes):
    pred = prob.argmax(axis=1)
    out = {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    onehot = np.eye(num_classes)[y_true]
    try:
        out["AUROC"] = float(roc_auc_score(onehot, prob, average="macro", multi_class="ovr"))
    except ValueError:
        out["AUROC"] = float("nan")
    try:
        out["AUPRC"] = float(average_precision_score(onehot, prob, average="macro"))
    except ValueError:
        out["AUPRC"] = float("nan")
    return out


def epoch(mode, data, net, optimizer, criterion, args, aug=True):
    net.to(args.device)
    net.train("train" in mode)
    loss_sum, n_sum = 0.0, 0
    probs, labels = [], []
    for X_true, y_true in data["data_loader"]:
        for inputaug in args.inputaug_list:
            X = Input_Augmentation(X_true.to(args.device), inputaug)
            y = y_true.long().to(args.device)
            X = maybe_train_aug(X, args, aug=aug)
            if "train" in mode:
                out = net(X)
            else:
                with torch.no_grad():
                    out = net(X)
            logits = logits_from_output(out)
            loss = criterion(logits, y)
            n_b = y.numel()
            loss_sum += float(loss.item()) * n_b
            n_sum += n_b
            probs.append(F.softmax(logits.detach(), dim=1).cpu())
            labels.append(y.detach().cpu())
            if "train" in mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    prob_np = torch.cat(probs).numpy()
    y_np = torch.cat(labels).numpy()
    return loss_sum / max(1, n_sum), metric_dict_from_arrays(y_np, prob_np, args.num_classes)


def epoch_origin(mode, data, net, optimizer, criterion, args, aug=False):
    saved = args.inputaug_list
    args.inputaug_list = ["raw"]
    try:
        return epoch(mode, data, net, optimizer, criterion, args, aug=aug)
    finally:
        args.inputaug_list = saved


def evaluate_synset(it_eval, net, real_data_dict, args, use_data="val", return_loss=False, aug=True):
    net = net.to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=float(args.lr_teacher), momentum=0.9, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    syn_loader = real_data_dict["syn"]
    test_loader = real_data_dict[use_data]
    train_loss = test_loss = 0.0
    test_metrics = {}
    time0 = time.time()
    for ep in range(int(args.epoch_eval_train) + 1):
        train_loss, train_metrics = epoch("train_evalsyn", {"data_loader": syn_loader, "model": args.model}, net, optimizer, criterion, args, aug=aug)
        if ep % 100 == 0 or ep == int(args.epoch_eval_train):
            log_verbose(args, f"Evaluation SynSet id : {it_eval}, Time : {time.time()-time0:.5f}\t Epoch: {ep}/{args.epoch_eval_train}\tTrain Loss : {train_loss:.6f}\t" + ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
        time0 = time.time()
        if ep == int(args.epoch_eval_train):
            test_loss, test_metrics = epoch_origin("test_evalsyn", {"data_loader": test_loader, "model": args.model}, net, None, criterion, args, aug=False)
            log_verbose(args, f"Evaluation SynSet id : {it_eval}, Epoch: {ep}\tTest Loss : {test_loss:.6f}\t" + ", ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))
    return (net, test_metrics, train_loss, test_loss) if return_loss else (net, test_metrics)


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == "S":
        return [model]
    if eval_mode == "M":
        return [model_eval]
    raise NotImplementedError(f"Unsupported eval_mode: {eval_mode}")
