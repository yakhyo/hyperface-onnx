# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from .hyperface_onnx import HyperFace
from .recognition import FaceRecognition

# ONNX inference is torch-free. The PyTorch reference net needs torch, so import
# it from the submodule directly: `from models.backbones import IResNet50`.

__all__ = ["FaceRecognition", "HyperFace"]
