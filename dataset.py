# =============================================================================
# Import required libraries
# =============================================================================
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np

'''
Sprite classes:
    0: human
    1: non-human
    2: food
    3: spell
    4: side-facing
'''


# =============================================================================
# Sprite data
# =============================================================================
class SpriteDataset(torch.utils.data.Dataset):
    def __init__(self, sfilename, lfilename, transform):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.sprites[idx]
        if self.transform:
            image = self.transform(image)
        label = np.argmax(self.slabels[idx])
        return image, label

    def __len__(self):
        return len(self.sprites)


# =============================================================================
# Make dataloader
# =============================================================================
def make_dataloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data = SpriteDataset(sfilename="./datasets/Sprites/sprites_1788_16x16.npy",
                         lfilename="./datasets/Sprites/sprite_labels_nc_1788_16x16.npy",
                         transform=transform)
    #
    dataloader = DataLoader(data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True)

    return dataloader
