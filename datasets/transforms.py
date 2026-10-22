"""Training image transforms with CLIP normalization."""
from __future__ import annotations

import random

from PIL import Image
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class RandomChannelAugment:
    """Reduce color bias while retaining pedestrian structure."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= 0.5:
            return image
        if random.random() < 0.5:
            return Image.merge("RGB", (random.choice(image.split()),) * 3)
        return transforms.functional.rgb_to_grayscale(image, num_output_channels=3)


def build_train_transform(height: int = 288, width: int = 144) -> transforms.Compose:
    """Apply resize, channel augmentation, tensor conversion and normalization."""
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be positive")
    return transforms.Compose(
        [
            transforms.Resize((height, width), antialias=True),
            RandomChannelAugment(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
