import glob
import os
import torch
import numpy as np
import random
import cv2
from torchvision.datasets import CIFAR10
from typing import Any, Callable, Optional, Tuple
from PIL import Image

class RotationLoader(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
        ):
        CIFAR10.__init__(self, root, train=train, transform=transform, download=download)

        
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        img = self.transform(img)
        img1 = torch.rot90(img, 1, [1,2])
        img2 = torch.rot90(img, 2, [1,2])
        img3 = torch.rot90(img, 3, [1,2])
        imgs = [img, img1, img2, img3]
        rotations = [0,1,2,3]
        random.shuffle(rotations)
        return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
