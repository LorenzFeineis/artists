import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms, utils

from create_dataset import ArtistsPaintingsDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from NeuralNet import Net

if torch.cuda.is_available():
  dev = "cuda:1"
else:
  dev = "cpu"
device = torch.device(dev)
print(device)

transform = transforms.Compose([transforms.CenterCrop(size=(256,256)),
                                transforms.ToTensor()])


#artists_idx = [8,13,15,16,19,20,22,30,31,32,46]
artists_idx = [8,15,20,30]

train_data = ArtistsPaintingsDataset(root_dir="aug_images/", transform = transform, mode="Train", artists_idx=artists_idx)
test_data = ArtistsPaintingsDataset(root_dir="aug_images/", transform = transform, mode="Test", artists_idx=artists_idx)
print(len(train_data), "train images loaded.")
print(len(test_data), "test images loaded.")

train_loader = DataLoader(train_data, batch_size=256)
test_loader = DataLoader(test_data)


net = Net(size= (256,256), num_classes = 4)
net.to(device)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = 1e-3, momentum = 0.9)

loss_train = []
loss_test = []

for epoch in range(1000):
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

loss_test = np.array(loss_test)
loss_train = np.array(loss_train)

np.save("test_loss_e-3.npy", loss_test)
np.save("train_loss_e-3.npy", loss_train)
torch.save(net, "model_lr_e-3.pt")
