#!/bin/bash

# Download hyperface-onnx model weights from GitHub releases
# Usage: ./download.sh

BASE_URL="https://github.com/yakhyo/hyperface-onnx/releases/download/weights"

WEIGHTS_DIR="weights"
mkdir -p ${WEIGHTS_DIR}

echo "Downloading model weights..."

echo "Downloading HyperFace IR-50 (hyperface_ir_50.onnx)..."
curl -L "${BASE_URL}/hyperface_ir_50.onnx" -o "${WEIGHTS_DIR}/hyperface_ir_50.onnx"

echo "Done! Weights saved to ${WEIGHTS_DIR}/"
ls -lh ${WEIGHTS_DIR}/
