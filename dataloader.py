import os
from glob import glob
from os.path import join

import imageio
import torch
import torchvision.transforms as transforms
from torch.utils import data


class CustomImageDataset(data.Dataset):
    """
    Custom Dataset for labeled images with 512x512 size.
    Handles dataset with subfolders representing labels.
    """

    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.filelist = []
        self.labels = []

        if self.split == "train":
            self.image_dir = join(self.root, "training")
        elif self.split == "test":
            self.image_dir = join(self.root, "testing")
        else:
            raise ValueError("Invalid split type! Use 'train' or 'test'.")

        self.label_folders = os.listdir(self.image_dir)

        for label_idx, label_folder in enumerate(self.label_folders):
            label_folder_path = join(self.image_dir, label_folder)
            image_files = glob(join(label_folder_path, "*.*"))

            self.filelist.extend(image_files)
            self.labels.extend([label_idx] * len(image_files))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file = self.filelist[index]
        img = imageio.imread(img_file)

        if img.shape[:2] != (512, 512):
            raise ValueError(
                f"Image size mismatch! Expected 512x512, but got {img.shape[:2]}"
            )

        if self.transform:
            img = self.transform(img)

        label = self.labels[index]

        return img, label
