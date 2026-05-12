from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class _RNNBase(nn.Module):
    _cell = nn.RNN

    def __init__(self, c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, fc_dropout=0.0):
        super().__init__()
        self.rnn = self._cell(c_in, hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.final_rep = hidden_size * (1 + bidirectional)
        self.fc = nn.Linear(self.final_rep, c_out)

    def embed(self, x):
        x = x.transpose(2, 1)
        output, _ = self.rnn(x)
        return output[:, -1]

    def forward(self, x):
        return self.fc(self.dropout(self.embed(x)))


class LSTM(_RNNBase):
    _cell = nn.LSTM


class GRU(_RNNBase):
    _cell = nn.GRU


class TransformerModel(nn.Module):
    def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="relu", n_layers=1):
        super().__init__()
        self.inlinear = nn.Linear(c_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model))
        self.relu = nn.ReLU()
        self.final_rep = d_model
        self.outlinear = nn.Linear(d_model, c_out)

    def embed(self, x):
        x = x.permute(2, 0, 1)
        x = self.relu(self.inlinear(x))
        x = self.transformer_encoder(x).permute(1, 0, 2)
        return self.relu(x.max(1, keepdim=False)[0])

    def forward(self, x):
        return self.outlinear(self.embed(x))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size else x


class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(nf, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(ni, nf, 1) if ni != nf else None
        self.relu = nn.ReLU()
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, c_in, c_out, layers=None, ks=7, conv_dropout=0.0, fc_dropout=0.0):
        super().__init__()
        layers = layers or [32, 32, 32, 32]
        blocks = []
        for i, nf in enumerate(layers):
            ni = c_in if i == 0 else layers[i - 1]
            dilation = 2**i
            blocks.append(TemporalBlock(ni, nf, ks, stride=1, dilation=dilation, padding=(ks - 1) * dilation, dropout=conv_dropout))
        self.tcn = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.final_rep = layers[-1]
        self.linear = nn.Linear(self.final_rep, c_out)
        self.linear.weight.data.normal_(0, 0.01)

    def embed(self, x):
        return self.dropout(self.gap(self.tcn(x)).reshape(x.shape[0], -1))

    def forward(self, x):
        return self.linear(self.embed(x))


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, hid2_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid2_dim)
        self.fc3 = nn.Linear(hid2_dim, out_dim)
        self.final_rep = hid2_dim

    def embed(self, x):
        x = F.relu(self.fc1(x.reshape(x.shape[0], -1)))
        return self.fc2(x)

    def forward(self, x):
        return self.fc3(F.relu(self.embed(x)))


class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_norm, im_size):
        super().__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, im_size)
        self.final_rep = shape_feat[0] * shape_feat[1]
        self.classifier = nn.Linear(self.final_rep, num_classes)

    def _norm(self, net_norm, shape_feat):
        if net_norm == "BN":
            return nn.BatchNorm1d(shape_feat[0], affine=True)
        if net_norm == "LN":
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        if net_norm == "IN":
            return nn.InstanceNorm1d(shape_feat[0], affine=True)
        if net_norm == "GN":
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        if net_norm == "none":
            return None
        raise ValueError(f"unknown net_norm: {net_norm}")

    def _make_layers(self, channel, net_width, net_depth, net_norm, im_size):
        layers, in_channels, shape_feat = [], channel, [channel, im_size]
        for _ in range(net_depth):
            layers.append(nn.Conv1d(in_channels, net_width, kernel_size=3, padding=1))
            shape_feat[0] = net_width
            norm = self._norm(net_norm, shape_feat)
            if norm is not None:
                layers.append(norm)
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            shape_feat[1] //= 2
            in_channels = net_width
        return nn.Sequential(*layers), shape_feat

    def embed(self, x):
        return self.features(x).view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self.embed(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm="instancenorm"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if norm == "instancenorm" else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if norm == "instancenorm" else nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(1, 1), stride=stride, bias=False),
                nn.GroupNorm(planes, planes, affine=True) if norm == "instancenorm" else nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm="instancenorm"):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if norm == "instancenorm" else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm=norm)
        self.pool = nn.AdaptiveAvgPool2d((4, 1))
        self.final_rep = 512 * block.expansion * 4
        self.classifier = nn.Linear(self.final_rep, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s, norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def embed(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.pool(out)
        return out.view(out.size(0), -1)

    def forward(self, x):
        return self.classifier(self.embed(x))


def get_network(args):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    model = args.model
    if model == "MLP" or model.lower() == "mlp":
        return MLP(args.time_step * args.channel, 128, 64 if getattr(args, "dual", 1) else 128, args.num_classes)
    if model in {"CNNIN", "CNNBN", "CNNLN"} or model.lower() == "cnn":
        norm = {"CNNIN": "IN", "CNNBN": "BN", "CNNLN": "LN"}.get(model, "IN")
        return ConvNet(args.channel, args.num_classes, 16 if getattr(args, "dual", 1) else 32, 3, norm, args.time_step)
    if model == "LSTM":
        return LSTM(args.channel, args.num_classes, hidden_size=64 if getattr(args, "dual", 1) else 100)
    if model == "GRU":
        return GRU(args.channel, args.num_classes, hidden_size=64 if getattr(args, "dual", 1) else 100)
    if model == "Transformer":
        return TransformerModel(args.channel, args.num_classes, d_model=32 if getattr(args, "dual", 1) else 64)
    if model == "TCN":
        return TCN(args.channel, args.num_classes, layers=[48] if getattr(args, "dual", 1) else [64])
    if model == "ResNet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], channel=args.channel, num_classes=args.num_classes, norm="instancenorm")
    if model == "ResNet18BN":
        return ResNet(BasicBlock, [2, 2, 2, 2], channel=args.channel, num_classes=args.num_classes, norm="batchnorm")
    raise NotImplementedError(f"Unsupported model: {model}")
