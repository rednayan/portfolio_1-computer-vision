# TODO: more augmentation needs to be experimented with!

import torchvision.transforms as T
import torch
import random
from typing import Tuple


class Compose:
    """A minimal Compose class to apply transforms sequentially."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:

        for t in self.transforms:
            if t.__class__.__name__ in ['RandomHorizontalFlip']:
                image, target = t(image, target)
            else:
                image = t(image)

        return image, target

class RandomHorizontalFlip:
    """Randomly flips the image and updates the bounding box coordinates."""
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        if random.random() < self.prob:
            image = T.functional.hflip(image)

            W = image.shape[2]

            boxes = target["boxes"]
            if boxes.numel() > 0:
                xmin_old = boxes[:, 0].clone()
                xmax_old = boxes[:, 2].clone()

                boxes[:, 0] = W - xmax_old
                boxes[:, 2] = W - xmin_old

                target["boxes"] = torch.stack((boxes[:, 0].clamp(min=0), boxes[:, 1], boxes[:, 2].clamp(max=W), boxes[:, 3]), 1)

        return image, target


def get_transform(train: bool) -> Compose:
    """
    Defines the complete transformation pipeline for the dataset.

    Args:
        train (bool): True for training set (includes augmentation), False for validation/test set.

    Returns:
        A Compose object containing the transforms.
    """
    transforms = []

    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1))

    return Compose(transforms)
