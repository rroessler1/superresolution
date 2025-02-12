import glob
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2, InterpolationMode


class SRDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        search_pattern = os.path.join(img_dir, "*.png")
        self.filenames = glob.glob(search_pattern)
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(64),
            v2.ColorJitter(0.2, 0.2, 0.2, 0.2)
        ])
        self.resize_transform = v2.Compose([
            v2.Resize(size=(32, 32), interpolation=InterpolationMode.BILINEAR)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = read_image(self.filenames[idx])
        img = self.transforms(img)
        downsampled_img = self.resize_transform(img)
        return downsampled_img, img


class SREvalDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        search_pattern = os.path.join(img_dir, "*.png")
        self.filenames = glob.glob(search_pattern)
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.resize_transform = v2.Compose([
            v2.Resize(size=(240, 240), interpolation=InterpolationMode.BILINEAR)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = read_image(self.filenames[idx])
        img = self.transforms(img)
        downsampled_img = self.resize_transform(img)
        return downsampled_img, img
