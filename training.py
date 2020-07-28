import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms, utils

from dataset import ArtistsPaintingsDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from NeuralNet import Net
import test_models

def training(name, batch_size = 256, lr=1e-4, num_epochs = 1000, cuda = 0, output = True):
    if torch.cuda.is_available():
      dev = "cuda:{}".format(cuda)
    else:
      dev = "cpu"
    device = torch.device(dev)
    print(device)

    transform = transforms.Compose([transforms.CenterCrop(size=(256,256)),
                                    transforms.ToTensor()])


    #artists_idx = [8,13,15,16,19,20,22,30,31,32,46]
    artists_idx = [8,15,20,30]

    train_data = ArtistsPaintingsDataset(transform = transform, mode="Train")
    test_data = ArtistsPaintingsDataset(ttransform = transform, mode="Test")
    print(len(train_data), "train images loaded.")
    print(len(test_data), "test images loaded.")

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data)


    net = Net(size= (256,256), num_classes = 4)
    net.to(device)

    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)

    loss_train = []
    loss_test = []

    for epoch in range(num_epochs):
        print("epoch:",epoch)
        for batch in tqdm(train_loader):
            x_train, y_train = batch
            y_train = y_train[0]

            optimizer.zero_grad()
            net.train()
            output = net(x_train.to(device))
            training_loss = loss(output.to(device), y_train.to(device))
            training_loss.backward()
            optimizer.step()
        print("training loss:", training_loss)
        loss_train.append(training_loss)
        net.eval()
        for batch in test_loader:
            x_test, y_test = batch
            y_test = y_test[0]
            output = net(x_test.to(device))
            test_loss = loss(output.to(device), y_test.to(device))
        print("test_loss:", test_loss)
        loss_test.append(test_loss)

        accuracy = [[],[]]
        if output and (epochs+1)%2 == 0:
            loss_test = np.array(loss_test)
            loss_train = np.array(loss_train)
            np.save(name+"_test_loss_{}.npy".format(str(lr)), loss_test)
            np.save(name+"_train_loss_{}.npy".format(str(lr)), loss_train)
            ####
            ####
            data = test_data, test_loader, train_data, train_loader
            train_accuracy, test_accuracy = performance(lr=lr,model = net, data=)
            accuracy[0].append(train_accuracy)
            accuracy[1].append(test_accuracy)
            np.save(name+"accuracy_lr_{},batch_{}.npy"format(str(lr),str(batch_size)), accuracy)
            ###
            ###
            ###

    loss_test = np.array(loss_test)
    loss_train = np.array(loss_train)
    if output:
        torch.save(net, name+"model_lr_{}_batch_{}.pt".format(str(lr),str(batch_size)))

if __name__ == "__main__":
    training(name = "TEST", cuda=1, batch_size=16, num_epochs=10, lr=1e-4, output = True)
