import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
# from torchvision.models.densenet import densenet121
import logging

import numpy as np


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ResNetEncoder(nn.Module):
    # ResNet 18 or 50 as image encoder
    def __init__(self, arch='resnet18', feature_dim=128):
        super(ResNetEncoder, self).__init__()

        self.f = []
        if arch == 'resnet50':
            backbone = resnet50()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'resnet18':
            backbone = resnet18()
            self.featdim = backbone.fc.weight.shape[1]

        for name, module in backbone.named_children():
            if name != 'fc':
                self.f.append(module)

        # encoder
        self.f = nn.Sequential(*self.f)

        # print(self.featdim)
        self.g = nn.Linear(self.featdim, feature_dim)

    def forward(self, x):
        feature = self.f(x)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.g(feature)
        return feature


class ConditionalModel(nn.Module):
    def __init__(self, config, n_steps, n_classes, guidance=False):
        super(ConditionalModel, self).__init__()
        logging.info("Initialize UNET")
        # n_steps = config["diffusion"]["timesteps"] + 1
        n_steps += 1
        # data_dim = config["model"]["data_dim"]
        # y_dim = config["data"]["num_classes"]
        y_dim = n_classes
        # arch = config["model"]["arch"]
        arch = config["arch"]
        # feature_dim = config["model"]["feature_dim"]
        feature_dim = config["feature_dim"]
        # hidden_dim = config["model"]["hidden_dim"]
        self.guidance = guidance
        # encoder for x
        self.encoder_x = ResNetEncoder(arch=arch, feature_dim=feature_dim)
        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat=None):
        # print(x.shape)
        x = self.encoder_x(x)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)
        if self.guidance:
            # for yh in yhat:
            y = torch.cat([y, yhat], dim=-1)
        # print(y.shape)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        y = self.lin4(y)

        return y
