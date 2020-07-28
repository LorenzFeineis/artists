import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms, utils

from dataset import ArtistsPaintingsDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm






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
    transform = transforms.Compose([transforms.CenterCrop(size=(256,256)),
                                    transforms.ToTensor()])
    if Test:
        test_data = ArtistsPaintingsDataset(transform = transform, mode="Test")
        test_loader = DataLoader(test_data,batch_size=1)

    if Train:
        train_data = ArtistsPaintingsDataset(transform = transform, mode="Train")
        train_loader = DataLoader(train_data)

    if Test and Train:
        return test_data, test_loader, train_data, train_loader
    elif Test:
        return test_data,test_loader, None, None
    elif Train:
        return train_data, train_loader, None,None



def performance(lr="e-5", load_model = None, data = "Load"):
    if data=="Load":
        test_data, test_loader, train_data, train_loader = load_data()
    else:
        test_data, test_loader, train_data, train_loader = data

    if load_model==None:
        model = torch.load("model_lr_{}.pt".format(lr))
    else:
        model = load_model

    model.eval()

    test_accuracy = 0
    for batch in test_loader:
        x_test, y_test = batch
        y_test = y_test[0]
        test_output = model(x_test)
        prediction = np.argmax(test_output.cpu().detach().numpy(),axis = 1)
        ground_truth = y_test.cpu().detach().numpy()
        test_accuracy += len(prediction)- np.count_nonzero(prediction-ground_truth)

    train_accuracy = 0
    for batch in train_loader:
        x_train, y_train = batch
        y_train = y_train[0]
        train_output = model(x_train)
        print("training output", train_output.cpu().detach().numpy())
        prediction = np.argmax(train_output.cpu().detach().numpy(),axis = 1)
        ground_truth = y_train.cpu().detach().numpy()
        print("prediction:", prediction)
        print("ground truth:", ground_truth)
        train_accuracy += len(prediction)- np.count_nonzero(prediction-ground_truth)


    return train_accuracy/len(train_data), test_accuracy//len(test_data)

if __name__ == "__main__":
    #plot_losses()
    performance(lr="e-5")
