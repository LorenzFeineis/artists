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









def plot_losses(lr="e-5"):
    test_loss = np.load("test_loss_{}.npy".format(lr),allow_pickle=True)
    train_loss = np.load("train_loss_{}.npy", allow_pickle=True)

    fig, axis = plt.subplots(ncols = 2)

    axis[0].plot(test_loss)
    axis[0].set(title = "test loss",xlabel="epochs", ylabel="CELoss")
    axis[1].plot(train_loss)
    axis[1].set(title = "trainings loss",xlabel="epochs", ylabel="CELoss")
    plt.savefig("losses_{}.png")

def performance(lr="e-5"):
    model = torch.load("model_lr_{}.pt".format(lr))
    model.eval()

    artists_idx = [8,15,20,30]

    #train_data = ArtistsPaintingsDataset(root_dir="aug_images/", transform = transform, mode="Train", artists_idx=artists_idx)
    test_data = ArtistsPaintingsDataset(root_dir="aug_images/", transform = transform, mode="Test", artists_idx=artists_idx)
    #print(len(train_data), "train images loaded.")
    print(len(test_data), "test images loaded.")

    #train_loader = DataLoader(train_data)
    test_loader = DataLoader(test_data,batch_size=len(test_data))

    #train_labels = torch.tensor(train_data.targets)
    test_labels = torch.tensor(test_data.targets)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    print(images.shape)
    test_output = model(images)
    print(test_output.shape)

if __name__ == "__main__":
    #plot_losses()
    performance(lr="e-4")
