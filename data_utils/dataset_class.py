import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import os
import pickle
import cv2
from typing import List, Dict, Any, Tuple
from data_utils.data_augmentation import get_transform

INPUT_FILE = 'formatted_annotations.pkl'
TRAIN_IDS_FILE = 'split_dataset/train_ids.txt'
VAL_IDS_FILE = 'split_dataset/val_ids.txt'
TEST_IDS_FILE = 'split_dataset/test_ids.txt'


def load_split_ids(file_path: str) -> List[str]:
    """Loads image IDs from a text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split ID file not found: {file_path}")
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_annotations(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads the full annotations and maps them by image_id for quick lookup."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Annotation file not found: {file_path}")
    with open(file_path, 'rb') as f:
        records = pickle.load(f)

    return {record['image_id']: record for record in records}


class WaterfowlDataset(Dataset):
    def __init__(self, split: str, transforms: T.Compose = None):
        self.transforms = transforms

        self.all_annotations = load_annotations(INPUT_FILE)

        if split == 'train':
            id_file = TRAIN_IDS_FILE
        elif split == 'val':
            id_file = VAL_IDS_FILE
        elif split == 'test':
            id_file = TEST_IDS_FILE
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        self.image_ids = load_split_ids(id_file)

        self.records = [self.all_annotations[id] for id in self.image_ids]

        print(f"Initialized WaterfowlDataset with {len(self.records)} images for split: {split}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Loads and preprocesses the image and its target annotations.
        """
        record = self.records[idx]

        img_path = record['file_path']
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise IOError(f"Failed to load image at {img_path}")

        if img.dtype != np.uint8:
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = 255 * (img.astype(np.float32) - img_min) / (img_max - img_min)

            img = img.astype(np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_tensor = T.functional.to_tensor(img_rgb)

        boxes = []
        labels = []

        if record['has_objects']:
            for ann in record['annotations']:
                boxes.append(ann['bbox'])
                labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
             img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))