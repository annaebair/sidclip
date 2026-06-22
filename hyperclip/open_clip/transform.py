import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        new_size = tuple(round(dim * scale) for dim in (height, width))
        if scale != 1.0:
            img = F.resize(img, new_size, self.interpolation)
        if not width == height:
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img

#import os
#import io
#import numpy as np
#from PIL import Image
#from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
#from cryptography.hazmat.primitives import padding
#from cryptography.hazmat.backends import default_backend

def generate_key():
    return os.urandom(32)  # AES-256 key

def encrypt(data, key):
    # Generate a random IV
    iv = os.urandom(16)

    # Create a Cipher object
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Pad data to be a multiple of block size
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()

    # Encrypt the padded data
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    return iv + ciphertext


# class ConvertToBytes:
#     def __call__(self, img):
#         img_array = np.array(img, dtype=np.uint8)
#         buffer = io.BytesIO()
#         Image.fromarray(img_array).save(buffer, format='JPEG')
#         buffer.seek(0)
#         img_bytes = list(buffer.read())

#         lenn = 24000
#         pad_length = max(0, lenn - len(img_bytes))
#         padding = [256] * pad_length
#         padded_img_bytes = np.concatenate([padding, img_bytes])

#         if len(padded_img_bytes)  > lenn:
#             padded_img_bytes = padded_img_bytes[:lenn]


#         return torch.tensor(padded_img_bytes).int()


class ConvertToBytes:
    def __call__(self, img):
        img_array = np.array(img, dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(img_array).save(buffer, format='JPEG')
        buffer.seek(0)
        img_bytes = buffer.read()

        self.key = generate_key()
        encrypted_bytes = encrypt(img_bytes, self.key)
        img_bytes_array = np.frombuffer(encrypted_bytes, dtype=np.uint8)

        # Define the target length
        lenn = 24000
        pad_length = max(0, lenn - len(img_bytes_array))
        padding = np.full(pad_length, 256, dtype=np.uint8)
        padded_img_bytes = np.concatenate([img_bytes_array, padding])

        if len(padded_img_bytes) > lenn:
            padded_img_bytes = padded_img_bytes[:lenn]

        return torch.tensor(padded_img_bytes).int()


def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    # import pdb; pdb.set_trace()
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = Normalize(mean=mean, std=std)
    use_bytes = False
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault('interpolation', 'random')

            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            print(aug_cfg_dict)
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                **aug_cfg_dict,
            )
        else:
            if use_bytes:
                train_transform = Compose([
                    RandomResizedCrop(
                        image_size,
                        scale=aug_cfg_dict.pop('scale'),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    ConvertToBytes()
                ])
                return train_transform
            # print(aug_cfg_dict)
            train_transform = Compose([
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
            if aug_cfg_dict:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        if use_bytes:
           print("use bytes")
           return Compose([
                    Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_size),
                    ConvertToBytes()
                ])
        elif resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


class ModelResizeTransform(nn.Module):
    def __init__(self, pretrained_model, image_size):
        super(ModelResizeTransform, self).__init__()
        self.pretrained_model = pretrained_model
        self.resize = ResizeMaxSize(image_size)

    def forward(self, x):
        with torch.no_grad():
            x = self.pretrained_model(x)

        x = self.resize(x)
        return x
