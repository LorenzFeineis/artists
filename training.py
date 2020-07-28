import torch
import numpy as np
from PIL import Image

import torchvision
from torchvision import transforms, utils

from dataset import ArtistsPaintingsDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from test_models import performance
from NeuralNet import Net

def training(name, batch_size = 256, lr=1e-4, num_epochs = 1000, cuda = 0, create_output = True):
    if torch.cuda.is_available():
      dev = "cuda:" + str(cuda)
    else:
      dev = "cpu"
    device = torch.device(dev)
    print(device)

    transform = transforms.ToTensor()

    artists_idx = [8,15,20,30]

    train_data = ArtistsPaintingsDataset(transform = transform, mode="Train")
    test_data = ArtistsPaintingsDataset(transform = transform, mode="Test")
    print(len(train_data), "train images loaded.")
    print(len(test_data), "test images loaded.")

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data)
    print("Do I load the data?")

    net = Net(size= (256,256), num_classes = 4,cuda = str(cuda))
    print("Problem after defining net")
    net.to(device)
    print("After sending net to device")
    loss = torch.nn.CrossEntropyLoss()
    print("after defining loss")
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
    print("after defining optimizer")
    loss_train = []
    loss_test = []

    print("Where am I?")
    for epoch in range(num_epochs):
        print("epoch:",epoch)
        train_loss = 0
        denominator = 0
        for batch in tqdm(train_loader):
            x_train, y_train = batch
            y_train = y_train[0]

            optimizer.zero_grad()
            net.train()
            output = net(x_train.to(device))
            training_loss = loss(output.to(device), y_train.to(device))
            train_loss += training_loss.data
            denominator += 1
            training_loss.backward()
            optimizer.step()

        train_loss = train_loss/denominator
        loss_train.append(train_loss)
        print("Mean training loss:", train_loss)

        net.eval()
        test_loss = 0
        for batch in test_loader:
            x_test, y_test = batch
            y_test = y_test[0]
            test_output = net(x_test.to(device))
            test_loss += loss(test_output.to(device), y_test.to(device)).data
        test_loss = test_loss/len(test_data)
        print("Mean test_loss:", test_loss)
        loss_test.append(test_loss)


        accuracy = [[],[]]
        if create_output:
            if (epoch+1)%2 == 0:
                np.save(name+"_test_loss_{}.npy".format(str(lr)), np.array(loss_test))
                np.save(name+"_train_loss_{}.npy".format(str(lr)), np.array(loss_train))

                data = test_data, test_loader, train_data, train_loader
                train_accuracy, test_accuracy = performance(lr=lr,load_model=net, data=data)
                print("Train accuracy:", train_accuracy)
                print("Test accuracy:", test_accuracy)
                accuracy[0].append(train_accuracy)
                accuracy[1].append(test_accuracy)
                np.save(name+"_accuracy_lr_{}_batch_{}.npy".format(str(lr),str(batch_size)), np.array(accuracy))
    if create_output:
        torch.save(net, name+"_model_lr_{}_batch_{}.pt".format(str(lr),str(batch_size)))

if __name__ == "__main__":
    training(name = "TEST", cuda=0, batch_size=16, num_epochs=4, lr=1e-4, create_output = True)
