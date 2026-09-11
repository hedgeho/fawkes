"""Face detection with InsightFace's SCRFD (det_10g), returning boxes and the 5 landmarks."""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DETECTOR_REPO = "public-data/insightface"
DETECTOR_FILE = "models/buffalo_l/det_10g.onnx"


def model_dir():
    """Directory holding downloaded weights: $FAWKES_MODEL_DIR or fawkes/model/ (gitignored)."""
    override = os.environ.get("FAWKES_MODEL_DIR")
    path = Path(override) if override else Path(__file__).resolve().parent / "model"
    path.mkdir(parents=True, exist_ok=True)
    return path


def detector_path():
    """Local path of det_10g.onnx, downloading it (17 MB) on first use."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(DETECTOR_REPO, DETECTOR_FILE, local_dir=model_dir()))


@dataclass
class DetectedFace:
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 in photo pixels
    kps: np.ndarray  # (5, 2): left eye, right eye, nose, left mouth, right mouth
    score: float

    @property
    def size(self):
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))


class Detector:
    """SCRFD-10GF via onnxruntime on CPU. `detect` takes an RGB uint8/float image."""

    def __init__(self, det_size=640, threshold=0.5, min_size=30):
        from insightface.model_zoo import get_model
        self.model = get_model(str(detector_path()), providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=-1, input_size=(det_size, det_size), det_thresh=threshold)
        self.min_size = min_size

    def detect(self, image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        bgr = np.ascontiguousarray(image[..., :3][..., ::-1]).astype(np.uint8)
        bboxes, kpss = self.model.detect(bgr, max_num=0, metric='default')
        faces = []
        for bbox, kps in zip(bboxes, kpss):
            face = DetectedFace(bbox=bbox[:4].astype(np.float32), kps=kps.astype(np.float32),
                                score=float(bbox[4]))
            if face.size >= self.min_size:
                faces.append(face)
        # largest first, so "the" face of a portrait is faces[0]
        faces.sort(key=lambda f: -f.size)
        return faces
