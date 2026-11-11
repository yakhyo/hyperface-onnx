# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import onnxruntime as ort
from uniface import face_alignment

__all__ = ["HyperFace"]


class HyperFace:
    """HyperFace face encoder, ONNX Runtime inference.

    The model is an IR-50 trained only on synthetic identities. It takes a
    112x112 aligned face in BGR order, as the AdaFace code base it was trained
    with does, and returns a 512-dimensional L2-normalised embedding.
    """

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """Load a HyperFace ONNX model.

        Args:
            model_path: Path to the ONNX file.
            providers: ONNX Runtime execution providers. Defaults to
                auto-detected (CUDA > CoreML > CPU).
        """
        if providers is None:
            available = ort.get_available_providers()
            preferred = [
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in preferred if p in available]

        options = ort.SessionOptions()
        options.log_severity_level = 3  # silence CoreML partition warnings
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (112, 112)

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Turn an aligned BGR face into the network input.

        Args:
            face_image: Aligned face in BGR format, shape (H, W, 3).

        Returns:
            Blob of shape (1, 3, 112, 112), BGR, scaled to [-1, 1].
        """
        resized = cv2.resize(face_image, self.input_size)
        blob = (resized.astype(np.float32) - 127.5) / 127.5
        return np.transpose(blob, (2, 0, 1))[np.newaxis]

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract the embedding of one face.

        Args:
            image: Full image in BGR format.
            landmarks: The face's five landmarks, shape (5, 2).

        Returns:
            The L2-normalised embedding, shape (512,).
        """
        aligned_face, _ = face_alignment(image, landmarks)
        embedding = self.session.run(
            None, {self.input_name: self.preprocess(aligned_face)}
        )[0]
        return embedding.ravel()
