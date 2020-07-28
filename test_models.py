import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms, utils

from create_dataset import ArtistsPaintingsDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


transform = transforms.Compose([transforms.CenterCrop(size=(256,256)),
                                transforms.ToTensor()])

artists_idx = [8,15,20,30]

#train_data = ArtistsPaintingsDataset(root_dir="aug_images/", transform = transform, mode="Train", artists_idx=artists_idx)
#test_data = ArtistsPaintingsDataset(root_dir="aug_images/", transform = transform, mode="Test", artists_idx=artists_idx)
#print(len(train_data), "train images loaded.")
#print(len(test_data), "test images loaded.")

#train_loader = DataLoader(train_data)
#test_loader = DataLoader(test_data)

model = torch.load("model_lr_e-4.pt")

test_loss = np.load("test_loss_e-4.npy",allow_pickle=True)
train_loss = np.load("train_loss_e-4.npy", allow_pickle=True)

def plot_losses():
    fig, axis = plt.subplot(ncols = 2)

    axis[0].plot(test_loss.values())
    axis[0].plot(train_loss.values())

    plt.savefig("losses_e-4.png")


if __name__ = "__main__":
    plot_losses()
