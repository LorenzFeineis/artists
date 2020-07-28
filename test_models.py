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


def load_data(Train=True, Test = True):
    if Test:
        test_data = ArtistsPaintingsDataset(root_dir="test_aug_images/", transform = transform, mode="Test", artists_idx=artists_idx)
        test_loader = DataLoader(test_data,batch_size=1)

    if Train:
    train_data = ArtistsPaintingsDataset(root_dir="train_aug_images/", transform = transform, mode="Train", artists_idx=artists_idx)
    train_loader = DataLoader(train_data)

    if Test and Train:
        return test_data, test_loader, train_data, train_loader
    elif Test:
        return test_data,test_loader, None, None
    elif Train:
        return train_data, train_loader, None,None



def performance(lr="e-5", model = None, data = "Load"):
    if data=="Load":
        test_data, test_loader, train_data, train_loader = load_data()
    else:
        data = data
    if load==None:
        model = torch.load("model_lr_{}.pt".format(lr))
    else:
        model = model

    model.eval()

    artists_idx = [8,15,20,30]

    test_accuracy = 0
    for batch in tqdm(test_loader):
        x_test, y_test = batch
        y_test = y_test[0]
        test_output = model(x_test)
        prediction = np.argmax(test_output.cpu().detach().numpy())
        ground_truth = y_test.cpu().detach().numpy()
        if ground_truth[0]==prediction:
            test_accuracy += 1

    train_accuracy = 0
    for batch in tqdm(train_loader):
        x_train, y_train = batch
        y_train = y_train[0]
        train_output = model(x_train)
        prediction = np.argmax(train_output.cpu().detach().numpy())
        ground_truth = y_train.cpu().detach().numpy()
        if ground_truth[0]==prediction:
            train_accuracy += 1

    return train_accuracy/len(test_data), test_accuracy//len(train_data)

if __name__ == "__main__":
    #plot_losses()
    performance(lr="e-5")
