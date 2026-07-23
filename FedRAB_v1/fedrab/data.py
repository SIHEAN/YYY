from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


CITYSCAPES_ID_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for source, target in {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}.items():
    CITYSCAPES_ID_TO_TRAINID[source] = target

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _scan_split(root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_root = root / "leftImg8bit" / split
    label_root = root / "gtFine" / split
    pairs = []
    for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
        city = image_path.parent.name
        stem = image_path.name.replace("_leftImg8bit.png", "")
        label = label_root / city / f"{stem}_gtFine_labelIds.png"
        if label.exists():
            pairs.append((image_path, label))
    if not pairs:
        raise FileNotFoundError(f"No Cityscapes {split} pairs found below {root}")
    return pairs


def _extract_client_entries(partition: Mapping) -> Dict[str, Sequence]:
    for key in ("clients", "client_data", "client_indices", "partitions"):
        value = partition.get(key)
        if isinstance(value, Mapping):
            return {str(k): v for k, v in value.items()}
    digit_keys = {str(k): v for k, v in partition.items() if str(k).isdigit()}
    if digit_keys:
        return digit_keys
    raise ValueError("Unsupported partition JSON: no client mapping was found")


def _unwrap_entries(value):
    if isinstance(value, Mapping):
        for key in ("train", "indices", "images", "samples", "data"):
            if key in value:
                return value[key]
    return value


def load_partition(data_root: str, partition_json: str) -> Dict[int, List[Tuple[Path, Path]]]:
    root = Path(data_root)
    all_pairs = _scan_split(root, "train")
    by_stem = {
        image.name.replace("_leftImg8bit.png", ""): (image, label)
        for image, label in all_pairs
    }
    payload = json.loads(Path(partition_json).read_text())
    raw_clients = _extract_client_entries(payload)
    result: Dict[int, List[Tuple[Path, Path]]] = {}
    for client_key, raw in raw_clients.items():
        entries = _unwrap_entries(raw)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"Client {client_key} entries are not a sequence")
        selected = []
        for item in entries:
            if isinstance(item, int):
                selected.append(all_pairs[item])
                continue
            text = str(item)
            stem = Path(text).name
            stem = stem.replace("_leftImg8bit.png", "").replace("_gtFine_labelIds.png", "")
            if stem not in by_stem:
                raise KeyError(f"Partition item not found in Cityscapes train split: {item}")
            selected.append(by_stem[stem])
        result[int(client_key)] = selected
    return result


class CityscapesFederatedDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        train: bool,
        crop_size: Tuple[int, int] = (512, 512),
        scale_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.pairs = list(pairs)
        self.train = train
        self.crop_size = crop_size
        self.scale_range = scale_range

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, index: int):
        image_path, label_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        raw = np.asarray(Image.open(label_path), dtype=np.uint8)
        label = Image.fromarray(CITYSCAPES_ID_TO_TRAINID[raw])
        return image, label, image_path.stem

    def __getitem__(self, index: int):
        image, label, sample_id = self._load(index)
        if self.train:
            scale = random.uniform(*self.scale_range)
            new_h = max(self.crop_size[0], int(round(image.height * scale)))
            new_w = max(self.crop_size[1], int(round(image.width * scale)))
            image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
            label = TF.resize(label, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)
            top = random.randint(0, max(0, new_h - self.crop_size[0]))
            left = random.randint(0, max(0, new_w - self.crop_size[1]))
            image = TF.crop(image, top, left, *self.crop_size)
            label = TF.crop(label, top, left, *self.crop_size)
            if random.random() < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
        image_tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
        label_tensor = torch.from_numpy(np.asarray(label, dtype=np.int64).copy()).long()
        return image_tensor, label_tensor, sample_id


def build_datasets(data_root: str, partition_json: str, crop_size=(512, 512)):
    root = Path(data_root)
    partition = load_partition(data_root, partition_json)
    client_sets = {
        client_id: CityscapesFederatedDataset(pairs, True, crop_size=crop_size)
        for client_id, pairs in partition.items()
    }
    val_pairs = _scan_split(root, "val")
    val_set = CityscapesFederatedDataset(val_pairs, False, crop_size=crop_size)
    return client_sets, val_set
