from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class MockSample:
    image: np.ndarray
    text_ids: np.ndarray
    struct_features: np.ndarray
    label: np.ndarray


class MockMultiModalDataset(Dataset):
    """用于联调的多模态mock数据集。"""

    def __init__(
        self,
        num_samples: int,
        image_size: int,
        text_max_len: int,
        num_struct_features: int,
        num_classes: int,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.samples: List[MockSample] = []

        for _ in range(num_samples):
            image = rng.normal(0.5, 0.2, size=(1, image_size, image_size)).astype(np.float32)
            text_ids = rng.integers(1, 999, size=(text_max_len,), dtype=np.int64)
            struct = rng.normal(0, 1, size=(num_struct_features,)).astype(np.float32)
            label = (rng.random(num_classes) > 0.7).astype(np.float32)
            if label.sum() == 0:
                label[rng.integers(0, num_classes)] = 1.0
            self.samples.append(MockSample(image=image, text_ids=text_ids, struct_features=struct, label=label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "image": torch.tensor(item.image, dtype=torch.float32),
            "text_ids": torch.tensor(item.text_ids, dtype=torch.long),
            "struct_features": torch.tensor(item.struct_features, dtype=torch.float32),
            "label": torch.tensor(item.label, dtype=torch.float32),
        }


def split_indices(num_samples: int, train_split: float) -> Tuple[List[int], List[int]]:
    pivot = int(num_samples * train_split)
    all_idx = list(range(num_samples))
    return all_idx[:pivot], all_idx[pivot:]


def tokenize_text_to_ids(text: str, max_len: int, vocab_size: int) -> np.ndarray:
    """将文本转成固定长度token id（轻量哈希分词占位）。"""
    tokens = text.strip().split()
    ids = np.zeros((max_len,), dtype=np.int64)
    for i, token in enumerate(tokens[:max_len]):
        ids[i] = (abs(hash(token)) % (vocab_size - 1)) + 1
    return ids


class CSVMultiModalDataset(Dataset):
    """
    真实数据接入数据集。
    CSV字段要求:
    image_path,text,struct_0..struct_n,label_0..label_m
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        image_size: int,
        text_max_len: int,
        num_struct_features: int,
        num_classes: int,
        text_vocab_size: int = 1000,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.text_max_len = text_max_len
        self.num_struct_features = num_struct_features
        self.num_classes = num_classes
        self.text_vocab_size = text_vocab_size

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, image_path: str) -> np.ndarray:
        img = Image.open(self.image_root / image_path).convert("L")
        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image = self._read_image(str(row["image_path"]))
        text_ids = tokenize_text_to_ids(str(row.get("text", "")), self.text_max_len, self.text_vocab_size)

        struct = np.array(
            [float(row[f"struct_{i}"]) for i in range(self.num_struct_features)],
            dtype=np.float32,
        )
        label = np.array(
            [float(row[f"label_{i}"]) for i in range(self.num_classes)],
            dtype=np.float32,
        )
        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "struct_features": torch.tensor(struct, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }
