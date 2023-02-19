from typing import *

import numpy as np

from docile.dataset import (
    BBox,
    Dataset as DocileRawDataset,
)
from torch.utils.data import Dataset
from transformers.utils.generic import PaddingStrategy
from transformers.tokenization_utils_fast import TruncationStrategy


DATASET_PATH = "docile/data/docile"


def normalize_bbox(
    bbox: BBox, page_size: tuple[int, int], max_value: int = 1024
) -> tuple[int, int, int, int]:
    return (
        bbox.to_relative_coords(*page_size)  # 0 ≤ coord ≤ 1
        .to_absolute_coords(max_value - 1, max_value - 1)  # 0 ≤ coord < max_value
        .to_tuple()
    )


class DocileDataset(Dataset):
    def __init__(self, split: str, processor):
        super().__init__()
        self._dataset = DocileRawDataset(split, DATASET_PATH)
        self.processor = processor

    def __len__(self) -> int:
        return len(self._dataset)

    @overload
    def __getitem__(self, idx: int) -> dict[str, Any]:
        ...

    @overload
    def __getitem__(self, idx: slice) -> list[dict[str, Any]]:
        ...

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            r = range(idx.start or 0, idx.stop, idx.step or 1)
            return [self[i] for i in r]

        doc = self._dataset[idx]

        image = doc.page_image(page=0)
        fields = doc.ocr.get_all_words(page=0)

        page_size = doc.page_image_size(0)
        bboxes = [
            normalize_bbox(field.bbox, page_size, max_value=1024) for field in fields
        ]
        words = [field.text for field in fields]

        encoding = self.processor(
            images=image,
            text=words,
            boxes=bboxes,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors="np",
        )

        return {
            k: (v.squeeze(0) if isinstance(v, np.ndarray) else v)
            for k, v in encoding.items()
        }
