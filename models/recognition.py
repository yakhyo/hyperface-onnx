# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
from uniface import RetinaFace, compute_similarity
from uniface.detection import BaseDetector

from .hyperface_onnx import HyperFace


class FaceRecognition:
    """High-level face recognition API combining a UniFace detector and the HyperFace encoder."""

    def __init__(
        self,
        recognition_path: str = "weights/hyperface_ir_50.onnx",
        detector: BaseDetector | None = None,
    ) -> None:
        """Load the encoder and a face detector.

        Args:
            recognition_path: Path to the HyperFace ONNX file.
            detector: Any UniFace detector that returns five landmarks. Defaults to
                `RetinaFace()`, whose weights UniFace downloads on first use.
        """
        self.detector = detector or RetinaFace()
        self.recognizer = HyperFace(model_path=recognition_path)

    def get_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """Extract the embedding of the most confident face. Returns None if no face is detected."""
        faces = self.detector.detect(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.confidence)
        return self.recognizer.get_embedding(image, face.landmarks)

    def compare(
        self, image1: np.ndarray | str, image2: np.ndarray | str
    ) -> float | None:
        """Compare two face images. Returns similarity score or None if detection fails."""
        if isinstance(image1, str):
            image1 = cv2.imread(image1)
        if isinstance(image2, str):
            image2 = cv2.imread(image2)

        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)

        if emb1 is None or emb2 is None:
            return None

        return float(compute_similarity(emb1, emb2, normalized=True))

    def is_match(
        self,
        image1: np.ndarray | str,
        image2: np.ndarray | str,
        threshold: float = 0.3,
    ) -> bool | None:
        """Check if two images contain the same person."""
        similarity = self.compare(image1, image2)
        if similarity is None:
            return None
        return similarity >= threshold
