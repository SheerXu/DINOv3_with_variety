from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from ..utils.label_map import build_label_map


class PolygonJsonDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        image_exts: List[str],
        label_map: Dict[str, int] | None = None,
        resize: Tuple[int, int] | None = None,
        random_flip: bool = False,
        ignore_index: int = 255,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_exts = [e.lower() for e in image_exts]
        self.resize = resize
        self.random_flip = random_flip
        self.ignore_index = ignore_index
        self.pairs = self._collect_pairs()

        if isinstance(label_map, list):
            label_map = {name: idx for idx, name in enumerate(label_map)}

        if label_map is None:
            labels = self._collect_labels()
            label_map = build_label_map(labels)

        self.label_map = label_map

    def _collect_pairs(self) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        for json_path in self.root_dir.rglob("*.json"):
            stem = json_path.stem
            image_path = None
            for ext in self.image_exts:
                candidate = json_path.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            pairs.append((image_path, json_path))
        return pairs

    def _collect_labels(self) -> List[str]:
        labels: List[str] = []
        for _, json_path in self.pairs:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label:
                    labels.append(label)
        return labels

    def __len__(self) -> int:
        return len(self.pairs)

    def _build_mask(self, data: Dict, size: Tuple[int, int]) -> Image.Image:
        w, h = size
        mask = Image.new("L", (w, h), color=0)
        drawer = ImageDraw.Draw(mask)
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in self.label_map:
                continue
            points = shape.get("points") or []
            if len(points) < 3:
                continue
            polygon = [(float(x), float(y)) for x, y in points]
            drawer.polygon(polygon, fill=self.label_map[label])
        return mask

    def __getitem__(self, index: int):
        image_path, json_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        mask = self._build_mask(data, image.size)

        if self.random_flip and torch.rand(1).item() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if self.resize is not None:
            image = F.resize(image, self.resize, interpolation=Image.BILINEAR)
            mask = F.resize(mask, self.resize, interpolation=Image.NEAREST)

        image_tensor = F.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor


def pad_collate(batch, ignore_index: int = 255):
    images, masks = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    padded_masks = []
    for img, mask in zip(images, masks):
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded_images.append(F.pad(img, [0, 0, pad_w, pad_h]))
        padded_masks.append(F.pad(mask, [0, pad_w, 0, pad_h], fill=ignore_index))

    return torch.stack(padded_images, dim=0), torch.stack(padded_masks, dim=0)
