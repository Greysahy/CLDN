# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
from PIL import ImageFilter
import random
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, k_transform):
        self.base_transform = base_transform
        self.k_transform = k_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.k_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
