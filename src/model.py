import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image

from . import config

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box: List[float]  # [x1, y1, x2, y2]


def load_class_meta() -> List[Dict]:
    names_list: List[str] = []
    if YOLO is not None and (config.MODEL_DIR / config.DEFAULT_MODEL).exists():
        try:
            yolo = YOLO(config.MODEL_DIR / config.DEFAULT_MODEL)
            names = yolo.model.names if hasattr(yolo, "model") else getattr(yolo, "names", {})
            names_list = [config.NAME_TRANSLATIONS.get(names[i], names[i]) for i in sorted(names)]  # type: ignore[index]
        except Exception:
            names_list = []
    if not names_list:
        names_list = config.FALLBACK_NAMES
    meta: List[Dict] = []
    for idx, name in enumerate(names_list):
        meta.append({"id": idx + 1, "name": name, "color": config.FALLBACK_COLORS[idx % len(config.FALLBACK_COLORS)]})
    return meta


CLASS_META = load_class_meta()


class ModelManager:
    """Manage YOLO model loading and inference."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Optional["YOLO"]] = {}

    def available(self) -> List[str]:
        return sorted([p.name for p in self.model_dir.glob("*.pt")])

    def get(self, name: str) -> Optional["YOLO"]:
        if name in self.cache:
            return self.cache[name]
        model_path = self.model_dir / name
        if YOLO is None or not model_path.exists():
            self.cache[name] = None
            return None
        try:
            self.cache[name] = YOLO(model_path)
        except Exception:
            self.cache[name] = None
        return self.cache[name]

    def predict(self, image: Image.Image, model_name: str) -> List[Detection]:
        model = self.get(model_name)
        if model is None:
            return self._mock_predict(image)
        results = model([image], imgsz=640, verbose=False)
        result = results[0]
        detections: List[Detection] = []
        names = getattr(model, "names", {}) or {}
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            cls_id = cls_idx + 1
            cls_name = self._class_name(cls_id, names.get(cls_idx, str(cls_idx)))
            conf = float(box.conf[0])
            if conf < config.CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection(cls_id=cls_id, cls_name=cls_name, conf=conf, box=[x1, y1, x2, y2]))
        return detections

    def _mock_predict(self, image: Image.Image) -> List[Detection]:
        width, height = image.size
        detections: List[Detection] = []
        for _ in range(random.randint(2, 4)):
            meta = random.choice(CLASS_META)
            w = width * random.uniform(0.18, 0.28)
            h = height * random.uniform(0.14, 0.24)
            x1 = random.uniform(0, width - w)
            y1 = random.uniform(0, height - h)
            detections.append(
                Detection(
                    cls_id=meta["id"],
                    cls_name=meta["name"],
                    conf=random.uniform(0.86, 0.98),
                    box=[x1, y1, x1 + w, y1 + h],
                )
            )
        return detections

    @staticmethod
    def _class_name(cls_id: int, default: str) -> str:
        meta = next((c for c in CLASS_META if c["id"] == cls_id), None)
        return meta["name"] if meta else default
