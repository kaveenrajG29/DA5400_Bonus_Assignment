from __future__ import annotations

import numpy as np
import torch
import torch.fft as fft


def Input_Augmentation(X, inputaug):
    if inputaug == "raw":
        return X
    if inputaug == "LPF":
        X_f = torch.fft.rfft(X, dim=-1)
        mask = torch.ones_like(X_f)
        mask[:, :, int(mask.shape[-1] * 0.5) :] = 0
        return torch.fft.irfft(X_f * mask, n=X.shape[-1], dim=-1)
    if inputaug == "FTPP":
        perturbation_factor = torch.pi / 4
        fft_data = fft.fft(X)
        phase_perturbation = torch.randn_like(fft_data) * perturbation_factor
        return fft.ifft(torch.abs(fft_data) * torch.exp(1j * (torch.angle(fft_data) + phase_perturbation))).real
    if inputaug == "FTMP":
        perturbation_factor = 0.01
        fft_data = fft.fft(X)
        amplitude = torch.abs(fft_data).clone()
        for c in range(amplitude.shape[1]):
            target = torch.mean(amplitude[:, c, 1:])
            amplitude[:, c, :] = amplitude[:, c, :] + torch.randn_like(amplitude[:, c, :]) * perturbation_factor * target
        return fft.ifft(amplitude * torch.exp(1j * torch.angle(fft_data))).real
    raise ValueError(f"unknown input augmentation: {inputaug}")


def jitter(x, args):
    return x + torch.normal(mean=0.0, std=args.jitter_ratio, size=x.shape, device=x.device)


def scaling(x, args):
    factor = torch.normal(mean=1.0, std=args.jitter_scale_ratio, size=(x.shape[0], x.shape[2]), device=x.device)
    return torch.cat([(x[:, i, :] * factor).unsqueeze(1) for i in range(x.shape[1])], dim=1)


def permutation(x, args, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, args.max_seg, size=x.shape[0])
    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points) if seg_mode == "random" else np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[:, warp]
        else:
            ret[i] = pat
    return ret


def fourier(sample, args):
    return torch.fft.rfft(sample, dim=-1)


def fourier_cat(sample, args):
    train_fft = torch.fft.rfft(sample, dim=-1)
    return torch.view_as_real(train_fft).reshape(train_fft.shape[0], train_fft.shape[1], -1)


def ifourier(sample, args):
    return torch.fft.irfft(sample, dim=-1)


def Aug_data(sample, args):
    aug = args.aug
    if aug in (None, "None"):
        return sample
    funcs = {
        "jitter": jitter,
        "scale": scaling,
        "scaling": scaling,
        "permutation": permutation,
        "fourier": fourier_cat if "fourier" in aug.split("_") and "ifourier" not in aug.split("_") else fourier,
        "ifourier": ifourier,
        "None": lambda x, _: x,
    }
    for name in aug.split("_"):
        sample = funcs[name](sample, args)
    return sample.real if torch.is_complex(sample) else sample


def DataTransform(sample, args):
    return scaling(sample, args), jitter(scaling(sample, args), args)


def maybe_train_aug(X, args, aug=True):
    if not aug:
        return X
    if args.aug in ["weak", "strong"]:
        weak_aug_X, strong_aug_X = DataTransform(X, args)
        return weak_aug_X if args.aug == "weak" else strong_aug_X
    if args.aug is not None:
        return Aug_data(X, args)
    return X
