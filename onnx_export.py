# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Export the PyTorch HyperFace net (models/backbones.py) to ONNX.

The .pt weights are the source of truth; this produces the ONNX the inference
pipeline runs. The export is sanity-checked against its PyTorch reference.

Dev-only requirements (not in requirements.txt): pip install torch onnx

Usage: python onnx_export.py
       python onnx_export.py --checkpoint HyperFace_50k_StyleGAN.ckpt   # start from Idiap's file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from models.backbones import IResNet50

WEIGHTS = Path(__file__).parent / "weights"
STATE_DICT = WEIGHTS / "hyperface_ir_50.pt"
TARGET = WEIGHTS / "hyperface_ir_50.onnx"

# AdaFace builds its network from unnamed Sequentials, so its tensors are called
# things like `body.3.res_layer.4.weight`. These tables give them the names used
# in models/backbones.py.
STEM = {"input_layer.0": "conv1", "input_layer.1": "bn1", "input_layer.2": "prelu"}
HEAD = {"output_layer.0": "bn2", "output_layer.3": "fc", "output_layer.4": "features"}
BLOCK = {
    "res_layer.0": "bn1",
    "res_layer.1": "conv1",
    "res_layer.2": "bn2",
    "res_layer.3": "prelu",
    "res_layer.4": "conv2",
    "res_layer.5": "bn3",
    "shortcut_layer.0": "downsample.0",
    "shortcut_layer.1": "downsample.1",
}
BLOCKS_PER_LAYER = (3, 4, 14, 3)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the PyTorch HyperFace net to ONNX"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Idiap's original Lightning checkpoint. If given, its backbone is saved to weights/ first",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def rename_key(key: str) -> str:
    """Map one AdaFace tensor name to its name in models/backbones.py."""
    if key.startswith("body."):
        # AdaFace numbers its 24 blocks in one flat list; here they sit in four layers.
        _, index, module, position, tensor = key.split(".")
        block = int(index)
        for layer, size in enumerate(BLOCKS_PER_LAYER, start=1):
            if block < size:
                return f"layer{layer}.{block}.{BLOCK[f'{module}.{position}']}.{tensor}"
            block -= size
        raise KeyError(key)

    module, tensor = key.rsplit(".", 1)
    return f"{(STEM | HEAD)[module]}.{tensor}"


def extract_backbone(checkpoint_path: Path) -> None:
    """Save the backbone of Idiap's training checkpoint as a plain state dict.

    The checkpoint also holds the optimizer state and the 50,000-way
    classification head used in training. Only the `model.*` tensors are the
    face encoder. They are renamed on the way, the values are untouched.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    prefix = "model."
    state = {
        rename_key(k[len(prefix) :]): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith(prefix)
    }
    # Fails loudly on any missing, unexpected or wrongly shaped tensor.
    IResNet50().load_state_dict(state)
    torch.save(state, STATE_DICT)
    print(
        f"{checkpoint_path.name} -> {STATE_DICT.name} ({STATE_DICT.stat().st_size / 1e6:.1f} MB)"
    )


@torch.no_grad()
def export(opset: int) -> None:
    net = IResNet50().eval()
    net.load_state_dict(torch.load(STATE_DICT, weights_only=True))

    dummy = torch.randn(1, 3, 112, 112)

    # Batch dim (dim 0) is dynamic on the input and the output, so several faces
    # can go through in one pass.
    torch.onnx.export(
        net,
        dummy,
        str(TARGET),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    # Sanity-check the export against the PyTorch reference before trusting it.
    session = ort.InferenceSession(str(TARGET), providers=["CPUExecutionProvider"])
    onnx_embedding = session.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(onnx_embedding - net(dummy).numpy()).max())
    print(
        f"{TARGET.name} ({TARGET.stat().st_size / 1e6:.1f} MB), vs PyTorch max |diff| = {max_diff:.2e}"
    )


def main() -> None:
    args = parse_arguments()
    if args.checkpoint:
        extract_backbone(args.checkpoint)
    export(args.opset)


if __name__ == "__main__":
    main()
