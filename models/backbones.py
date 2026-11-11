# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import torch
import torch.nn as nn

__all__ = ["IResNet", "IResNet50"]

# HyperFace was trained with the AdaFace code base
# (https://github.com/mk-minchul/AdaFace). Its IR-50 is InsightFace's IResNet-50
# with two differences, both kept here, so InsightFace weights do not load:
#   1. The first block of layer1 changes resolution but not width, and its
#      shortcut is plain subsampling with no weights. InsightFace puts a 1x1
#      conv and a BatchNorm there.
#   2. The final BatchNorm1d has no weight or bias.


class IBasicBlock(nn.Module):
    """Pre-activation 3x3-3x3 residual block used by IResNet."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        return out + identity


class IResNet(nn.Module):
    """Improved ResNet (AdaFace flavor). Input must be 112x112, BGR, scaled to [-1, 1].

    Returns the L2-normalised embedding with shape (N, num_features).
    """

    fc_scale = 7 * 7

    def __init__(self, layers: list[int], num_features: int = 512) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05, affine=False)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        elif stride != 1:
            # Same width, lower resolution: a 1x1 max-pool with a stride just subsamples.
            downsample = nn.MaxPool2d(1, stride)
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        x = self.fc(x)
        x = self.features(x)

        x = nn.functional.normalize(x, p=2, dim=1)

        return x


class IResNet50(IResNet):
    """IResNet-50 backbone of the released HyperFace models (43.6M parameters)."""

    def __init__(self) -> None:
        super().__init__(layers=[3, 4, 14, 3])
