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
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print(device)

transform = transforms.Compose([transforms.CenterCrop(size=(256,256)),
                                transforms.ToTensor()])


#artists_idx = [8,13,15,16,19,20,22,30,31,32,46]
artists_idx = [1]

train_data = ArtistsPaintingsDataset(transform = transform, mode="Train", artists_idx=artists_idx)
test_data = ArtistsPaintingsDataset(transform = transform, mode="Test", artists_idx=artists_idx)
print(len(train_data), "train images loaded.")
print(len(test_data), "test images loaded.")

train_loader = DataLoader(train_data, batch_size=2)
test_loader = DataLoader(test_data)


net = Net(size= (256,256), num_classes = 1)
net.to(device)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)

for epoch in range(5):
    print("epoch:",epoch)
    for batch in tqdm(train_loader):
        x_train, y_train = batch
        x_train.to(device)
        y_train = y_train[0]
        y_train.to(device)

        optimizer.zero_grad()
        net.train()
        output = net(x_train)
        output.to(device)
        training_loss = loss(output.to(device), y_train.to(device))
        training_loss.backward()
        optimizer.step()
    print(training_loss)
